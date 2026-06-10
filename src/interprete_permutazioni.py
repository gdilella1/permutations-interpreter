import math
import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Union, Callable, cast
from functions import Operator

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.animation import FuncAnimation

from lark import Lark, Tree, Token
from lark.lexer import Token as LarkToken
from lark.exceptions import UnexpectedToken, UnexpectedCharacters, LarkError

BUILTIN_NAMES: set[str] = set()
USER_VARS: list[str] = []
permutation_builtins = [
        ("identity",     ["n"],        "make_identity"),
        ("inverse",      ["p"],        "make_inverse"),
        ("power",        ["p","n"],    "make_power"),
        ("conjugate",    ["q","s"],    "make_conjugate"),
        ("order",        ["p"],        "make_order"),
        ("parity",       ["p"],        "make_parity"),
        ("fix",          ["p"],        "make_fix"),
        ("cyclesNumber", ["p"],        "make_cyclesNumber"),
        ("shift",        ["p","n"],    "make_shift"),
    ]

"""----------------------------------
        TIPI DI NODI AST
----------------------------------"""
@dataclass
class Loc:
    address: int
    
@dataclass
class Cycle:
    elements: List[int]

@dataclass
class CycleList:
    cycles: List[Cycle]
    
@dataclass
class Number:
    value: int

@dataclass
class Assign:
    name: str
    expression: 'Expression'
    
@dataclass
class VarDecl:
    name: str
    expression: "Expression"

@dataclass
class Print:
    expression: "Expression"

@dataclass
class CommandSequence:
    first: "Command"
    rest: Optional["CommandSequence"] = None

@dataclass
class BraidDiagram:
    name: str

@dataclass
class ListCmd:
    pass

@dataclass
class CyclesDiagram:
    name: str

@dataclass
class Let:
    name: str
    expression: "Expression"
    body: "Expression"

@dataclass
class Help:
    topic: Optional[str] = None
    
@dataclass
class Clear:
    pass

@dataclass
class Var:
    name: str
    
@dataclass
class Bool:
    value: bool

@dataclass
class Apply:
    op: str
    args: List['Expression']

@dataclass
class IfElse:
    cond: 'Expression'
    then_branch: 'CommandSequence'
    else_branch: 'CommandSequence'

@dataclass
class While:
    cond: 'Expression'
    body: 'CommandSequence'

@dataclass
class FunctionApp:
    name: str
    args: List["Expression"]
    
@dataclass
class Lambda:
    params: List[str]
    body: "Expression"

@dataclass
class Closure:
    params: List[str]
    body: "Expression"
    env: "Environment"
    
@dataclass
class EvalCommand:
    expression: "Expression"
    
Command = Assign | VarDecl | Print | BraidDiagram | CyclesDiagram | ListCmd | Help | Clear | IfElse | While | EvalCommand

Expression = CycleList | FunctionApp | Var | Lambda | Bool | Number | Apply

EVal = Union[int, bool, List[int]]
MVal = EVal
DVal = Union[MVal, Loc, Closure]


"""----------------------------------
        STATE + ENVIRONMENT
----------------------------------"""

@dataclass
class State:
    store: Callable[[int], MVal]
    next_loc: int

Environment = Callable[[str], DVal]

def empty_environment() -> Environment:
    def env_fn(name: str) -> DVal:
        raise ValueError(f"Variabile indefinita: {name}")
    return env_fn

def bind(env: Environment, name: str, value: DVal) -> Environment:
    def new_env(n: str) -> DVal:
        if n == name:
            return value
        return env(n)
    return new_env

def lookup(env: Environment, name: str) -> DVal:
    return env(name)
    
def empty_store() -> Callable[[int], MVal]:
    def store_fn(l: int) -> MVal:
        raise ValueError(f"Location {l} not allocated")
    return store_fn

def empty_state() -> State:
    return State(store=empty_store(), next_loc=0)

def allocate(state: State, value: MVal) -> tuple[Loc, State]:
    loc = Loc(state.next_loc)
    prev_store = state.store
    def new_store(l: int) -> MVal:
        if l == loc.address:
            return value
        return prev_store(l)
    return loc, State(store=new_store, next_loc=loc.address + 1)

def update(state: State, loc: Loc, value: MVal) -> State:
    prev_store = state.store
    def new_store(l: int) -> MVal:
        if l == loc.address:
            return value
        return prev_store(l)
    return State(store=new_store, next_loc=state.next_loc)

def access(state: State, loc: Loc) -> MVal:
    return state.store(loc.address)

"""----------------------------------
        ELABORAZIONE NODI AST
----------------------------------"""

def transform_expression(tree: Union[Tree, LarkToken]) -> Expression:
    if isinstance(tree, LarkToken):
        if tree.type == "IDENTIFIER":
            return Var(name=tree.value)
        if tree.type == "NUMBER":
            return Number(value=int(tree.value))
        if tree.type == "BOOL":
            return Bool(value=(tree.value == "true"))
        raise ValueError(f"Token inaspettato: {tree}")
  
    match tree.data:
        case "funapp":
            fname_tok, arg_list = tree.children
            args = []
            if isinstance(arg_list, Tree):
                args = [transform_expression(ch) for ch in arg_list.children]
            elif isinstance(arg_list, (Tree, LarkToken)):
                args = [transform_expression(arg_list)]
            return FunctionApp(name=fname_tok.value, args=args)

        case "cycle_list":
            return CycleList(cycles=[transform_expression(c) for c in tree.children])
        case "cycle":
            num_list = tree.children[0]
            elems = [int(tok.value) for tok in num_list.children if isinstance(tok, LarkToken)]
            return Cycle(elements=elems)

        case "lambda_expr":
            params_tree, body_tree = tree.children
            params = [tok.value for tok in (params_tree.children if params_tree else [])]
            body = transform_expression(body_tree)
            return Lambda(params=params, body=body)

        case "bin":
            left_tree, op_tok, right_tree = tree.children
            left  = transform_expression(left_tree)
            right = transform_expression(right_tree)
            return Apply(op=op_tok.value, args=[left, right])

        case "mono":
            children = tree.children
            if len(children) == 2 and isinstance(children[0], LarkToken) and children[0].type == "UNOP":
                op    = children[0].value
                arg   = transform_expression(children[1])
                return Apply(op=op, args=[arg])
            return transform_expression(children[0])
        
        case "let_expr":
            name_tok, expr_tree, body_tree = tree.children
            return Let(
                name=name_tok.value,
                expression=transform_expression(expr_tree),
                body=transform_expression(body_tree),
            )

        case "atom":
            return transform_expression(tree.children[0])

        case _:
            raise ValueError(f"Espressione non riconosciuta: {tree}")

def transform_command_tree(tree: Tree) -> Command:
    if tree.data == "assign":
        name_token, expr_tree = tree.children
        return Assign(
            name=name_token.value,
            expression=transform_expression(expr_tree)
        )
    elif tree.data == "vardecl":
        name_token, expr_tree = tree.children
        return VarDecl(
            name=name_token.value,
            expression=transform_expression(expr_tree)
        )
    elif tree.data == "print_cmd":
        (expr_tree,) = tree.children
        return Print(expression=transform_expression(expr_tree))
    elif tree.data == "clear_cmd":
        return Clear()
    elif tree.data == "list_cmd":
        return ListCmd()
    elif tree.data == "braid_diagram":
        (name_token,) = tree.children
        return BraidDiagram(name=name_token.value)
    elif tree.data == "cycles_diagram":
        (name_token,) = tree.children
        return CyclesDiagram(name=name_token.value)
    elif tree.data == "ifelse":
        cond_tree, then_tree, else_tree = tree.children
        return IfElse(
            transform_expression(cond_tree),
            transform_command_seq_tree(then_tree),
            transform_command_seq_tree(else_tree)
        )
    elif tree.data == "while":
        cond_tree, body_tree = tree.children
        return While(
            transform_expression(cond_tree),
            transform_command_seq_tree(body_tree)
        )
    elif tree.data == "help_cmd":
        if tree.children:
            (topic_token,) = tree.children
            return Help(topic=topic_token.value)
        else:
            return Help()
    elif tree.data == "funapp_cmd":
        (fun_tree,) = tree.children
        return EvalCommand(transform_expression(fun_tree))
    else:
        raise ValueError(f"transform_command_tree: nodo di comando non riconosciuto '{tree.data}'")

def transform_command_seq_tree(tree: Tree) -> CommandSequence:
    
    if tree.data != "command_seq":
        raise ValueError(f"transform_command_seq_tree si aspettava 'command_seq', ha ricevuto '{tree.data}'")
    children = [c for c in tree.children if isinstance(c, Tree)]
    if not children:
        raise ValueError("transform_command_seq_tree: 'command_seq' senza comandi all’interno.")
    if len(children) == 1:
        first_cmd = transform_command_tree(children[0])
        return CommandSequence(first=first_cmd, rest=None)
    first_cmd = transform_command_tree(children[0])
    rest_tree = Tree('command_seq', children[1:])
    rest_seq = transform_command_seq_tree(rest_tree)
    return CommandSequence(first=first_cmd, rest=rest_seq)

"""----------------------------------
        FUNZIONI DI APPOGGIO
----------------------------------"""

def anima_treccia_morph(p2: List[int], result: List[int], nome: str, p2_name: str):
    n = max(len(p2), len(result))
    p2 += list(range(len(p2) + 1, n + 1))
    result += list(range(len(result) + 1, n + 1))

    fig, ax = plt.subplots()
    ax.set_title(f"Morphing della treccia di '{nome}' a partire da quella di '{p2_name}'")
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(0, n + 1)
    ax.axis("off")

    for i in range(n):
        ax.text(-0.1, i + 1, str(i + 1), ha='right', va='center', fontsize=8)
        ax.text(1.1, result[i], str(result[i]), ha='left', va='center', fontsize=8)

    lines = [ax.plot([], [], 'k-')[0] for _ in range(n)]

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def interpolate(t, a, b):
        return (1 - t) * np.array(a) + t * np.array(b)

    def get_coords(mapping):
        return [(0, i + 1, 1, mapping[i]) for i in range(n)]

    coords_start = get_coords(p2)
    coords_end = get_coords(result)

    def animate(frame):
        t = frame / 60
        coords = [interpolate(t, (x0, y0, x1, y1), (x2, y2, x3, y3))
                  for (x0, y0, x1, y1), (x2, y2, x3, y3) in zip(coords_start, coords_end)]
        for i, (x0, y0, x1, y1) in enumerate(coords):
            lines[i].set_data([x0, x1], [y0, y1])
        return lines

    ani = FuncAnimation(fig, animate, init_func=init, frames=60, interval=40, blit=True, repeat=False)
    plt.show()

def mostra_diagramma(mapping: List[int], nome: str):
    n = len(mapping)
    fig, ax = plt.subplots()
    ax.set_title(f"Diagramma a treccia di '{nome}'")
    ax.axis("off")

    for i in range(n):
        ax.plot([0, 1], [i + 1, mapping[i]], 'k-')
        ax.plot(0, i + 1, 'ko')
        ax.text(-0.1, i + 1, str(i + 1), ha='right', va='center', fontsize=8)
        ax.plot(1, mapping[i], 'ko')
        ax.text(1.1, mapping[i], str(mapping[i]), ha='left', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()
    
def ciclo_to_perm(ciclo: List[int]) -> List[int]:
    n = max(ciclo)
    perm = list(range(1, n + 1))
    k = len(ciclo)
    for i in range(k):
        src = ciclo[i % k] - 1
        dst = ciclo[(i + 1) % k] - 1
        perm[src] = ciclo[(i + 1) % k]
    return perm

def permutazione_a_lista(lista: List[Cycle]) -> List[int]:
    max_val = max((e for ciclo in lista for e in ciclo.elements), default=0)
    perm = list(range(1, max_val + 1))

    for ciclo in reversed(lista):
        ciclo_perm = ciclo_to_perm(ciclo.elements)
        perm = prodotto_permutazioni(ciclo_perm, perm)

    return perm

def scrittura_in_cicli(perm: List[int]) -> str:
    n = len(perm)
    visited = [False] * n
    cicli = []

    for i in range(n):
        if not visited[i]:
            ciclo = []
            j = i
            while not visited[j]:
                visited[j] = True
                ciclo.append(j + 1)
                j = perm[j] - 1
            if len(ciclo) >= 1:
                cicli.append(f"({' '.join(map(str, ciclo))})")

    return ''.join(cicli) if cicli else "()"

def prodotto_permutazioni(p1: List[int], p2: List[int]) -> List[int]:
    p1_copy = p1.copy()
    p2_copy = p2.copy()
    n = max(len(p1_copy), len(p2_copy))
    p1_copy.extend(range(len(p1_copy) + 1, n + 1))
    p2_copy.extend(range(len(p2_copy) + 1, n + 1))
    return [p1_copy[p2_copy[i] - 1] for i in range(n)]

def mostra_cycles_diagramma(perm: List[int], nome: str):
    G = nx.DiGraph()
    n = len(perm)
    visited = [False] * n
    colori = list(mcolors.TABLEAU_COLORS.values())
    colore_idx = 0

    for i in range(n):
        if not visited[i]:
            ciclo = []
            j = i
            while not visited[j]:
                visited[j] = True
                ciclo.append(j)
                j = perm[j] - 1
            if ciclo:
                col = colori[colore_idx % len(colori)]
                colore_idx += 1
                for k in range(len(ciclo)):
                    src = ciclo[k] + 1
                    dst = perm[ciclo[k]]
                    G.add_edge(src, dst, color=col)

    pos = nx.circular_layout(G)
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]

    plt.figure()
    nx.draw_networkx_nodes(G, pos, node_color='white', edgecolors='black')
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True)
    plt.title(f"Diagramma a cicli di '{nome}'")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def conta_cicli(perm: List[int]) -> int:
    n = len(perm)
    visited = [False] * n
    ncicli = 0

    for i in range(n):
        if not visited[i]:
            ciclo = []
            j = i
            while not visited[j]:
                visited[j] = True
                ciclo.append(j + 1)
                j = perm[j] - 1
            if len(ciclo)>1:
                ncicli = ncicli + 1
    return ncicli
    
"""----------------------------------
                PARSER
----------------------------------"""

grammar = r"""
    program: command_seq

    command_seq: command (";" command)* ";"?

    ?command: assign
            | vardecl
            | print_cmd
            | clear_cmd
            | list_cmd
            | help_cmd
            | braid_diagram
            | cycles_diagram
            | ifelse
            | while
            | funapp_cmd

    assign: IDENTIFIER "<-" expression
    vardecl: "var" IDENTIFIER "=" expression

    print_cmd: "print" "(" expression ")"
    clear_cmd: "clear"
    list_cmd: "list"
    help_cmd: "help" IDENTIFIER?
    braid_diagram: "braidDiagram" "(" IDENTIFIER ")"
    cycles_diagram: "cyclesDiagram" "(" IDENTIFIER ")"

    ifelse: "if" expression "then" command_seq "else" command_seq "endif"
    while:  "while" expression "do" command_seq "done"

    funapp_cmd: funapp

    ?expression: lambda_expr
           | let_expr
           | bin
           | mono

    let_expr: "let" IDENTIFIER "=" expression "in" expression

    mono: atom
        | UNOP mono

    bin: expression OP mono

    atom: NUMBER
         | BOOL
         | IDENTIFIER
         | "(" expression ")"
         | cycle_list
         | funapp

    funapp: IDENTIFIER "(" [arg_list] ")"
    arg_list: expression ("," expression)*

    cycle_list: cycle+
    cycle: "(" number_list ")"
    number_list: NUMBER+

    lambda_expr: "lambda" "(" [param_list] ")" expression
    param_list: IDENTIFIER ("," IDENTIFIER)*

    OP: "==" | "!=" | "<=" | ">=" | "+" | "-" | "*" | "/" | "%" | "<" | ">" | "and" | "or"
    UNOP: "-" | "not"

    IDENTIFIER: /[a-zA-Z_][a-zA-Z0-9_]*/
    NUMBER:      /\d+/
    BOOL:        "true" | "false"

    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS
"""

def parse_program(program_text: str) -> CommandSequence:
    parse_tree = parser.parse(program_text)
    cmd_seq_tree = parse_tree.children[0]
    if not isinstance(cmd_seq_tree, Tree) or cmd_seq_tree.data != "command_seq":
        raise ValueError(f"[ERRORE PARSER] atteso 'command_seq', trovato {cmd_seq_tree}")
    return transform_command_seq_tree(cmd_seq_tree)

parser = Lark(grammar, start="program")

"""----------------------------------
            EVALUATOR
----------------------------------"""

def evaluate_expr(node, env: Environment, state: State) -> EVal:
    match node:
        case CycleList(cycles=clist):
            perm = permutazione_a_lista(clist)
            return perm
        case Cycle(elements=elems):
            return elems
        case FunctionApp(name="make_identity", args=[arg_n, _]):
            val_n = evaluate_expr(arg_n, env, state)
            if isinstance(val_n, int):
                size = val_n
            else:
                raise TypeError(f"identity richiede un intero, ho ottenuto: {val_n!r}")
            return list(range(1, size + 1))
        case FunctionApp(name="make_inverse", args=[arg_p, _]):
            perm = evaluate_expr(arg_p, env, state)
            n = len(perm)
            inv = [0] * n
            for i in range(n):
                inv[perm[i] - 1] = i + 1
            return inv
        case FunctionApp(name="make_power", args=[arg_p, arg_n, _]):
            base = evaluate_expr(arg_p, env, state)
            exp = int(evaluate_expr(arg_n, env, state))
            res = list(range(1, len(base) + 1))
            for _ in range(exp):
                res = prodotto_permutazioni(base, res)
            return res
        case FunctionApp(name="make_conjugate", args=[arg_q, arg_s, _]):
            p_q = evaluate_expr(arg_q, env, state)
            p_s = evaluate_expr(arg_s, env, state)
            n = max(len(p_q), len(p_s))
            qinv = [0] * n
            for i in range(len(p_q)):
                qinv[p_q[i] - 1] = i + 1
            for i in range(n):
                if qinv[i] == 0:
                    qinv[i] = i + 1
            if len(p_s) < n:
                p_s += list(range(len(p_s) + 1, n + 1))
            tmp = prodotto_permutazioni(p_s, qinv)
            return prodotto_permutazioni(p_q, tmp)
        case FunctionApp(name="make_order", args=[arg_p, _]):
            perm = evaluate_expr(arg_p, env, state)
            n = len(perm)
            visited = [False] * n
            lengths = []
            for i in range(n):
                if not visited[i] and perm[i] != i + 1:
                    length = 0
                    j = i
                    while not visited[j]:
                        visited[j] = True
                        j = perm[j] - 1
                        length += 1
                    if length > 0:
                        lengths.append(length)
            return math.lcm(*lengths) if lengths else 1
        case FunctionApp(name="make_parity", args=[arg_expr, _]):
            perm = evaluate_expr(arg_expr, env, state)
            if not isinstance(perm, list):
                raise TypeError(f"parity richiede una permutazione, ho ottenuto: {perm}")
            inv_count = sum(
                1
                for i in range(len(perm))
                for j in range(i + 1, len(perm))
                if perm[i] > perm[j]
            )
            return (inv_count % 2 == 0)
        case FunctionApp(name="make_fix", args=[arg_p, _]):
            perm = evaluate_expr(arg_p, env, state)
            fixed = [i + 1 for i, val in enumerate(perm) if val == i + 1]
            return fixed
        case FunctionApp(name="make_cyclesNumber", args=[arg_p, _]):
            perm = evaluate_expr(arg_p, env, state)
            return conta_cicli(perm)
        case FunctionApp(name="make_shift", args=[arg_p, arg_n, _]):
            perm = evaluate_expr(arg_p, env, state)
            if not isinstance(perm, list):
                raise TypeError(f"shift richiede una permutazione, ho ottenuto: {perm!r}")
            val_n = evaluate_expr(arg_n, env, state)
            if isinstance(val_n, int) and val_n>=0:
                offset = val_n
            else:
                raise TypeError(f"shift richiede un intero positivo come secondo argomento, ho ottenuto: {val_n!r}")
            return [k+1 for k in range(offset)]+[e + offset for e in perm]
        case FunctionApp(name="productVisualize", args=args):
            if len(args) < 2:
                raise ValueError("productVisualize richiede almeno due permutazioni.")
            perms: List[List[int]] = []
            names: List[str] = []
            for arg in args:
                perm = evaluate_expr(arg, env, state)
                if not isinstance(perm, list):
                    raise TypeError("productVisualize richiede permutazioni.")
                perms.append(perm.copy())
                names.append(arg.name if isinstance(arg, Var) else str(arg))
            current, current_name = perms[0], names[0]
            for next_perm, next_name in zip(perms[1:], names[1:]):
                composed = prodotto_permutazioni(current, next_perm)
                anima_treccia_morph(
                    current.copy(),
                    composed.copy(),
                    f"{current_name}*{next_name}",
                    current_name
                )
                current = composed
                current_name = f"{current_name}*{next_name}"
            return current
        case FunctionApp(name=fn_name, args=fn_args):
            dval = lookup(env, fn_name)
            if not isinstance(dval, Closure):
                raise ValueError(f"'{fn_name}' non è una funzione")
            closure = dval
            if len(closure.params) != len(fn_args):
                raise ValueError(
                    f"Funzione '{fn_name}' si aspetta {len(closure.params)} argomenti, "
                    f"ne ha ricevuti {len(fn_args)}"
                )
            arg_vals = [evaluate_expr(a, env, state) for a in fn_args]
            new_env = closure.env
            for pname, pval in zip(closure.params, arg_vals):
                if isinstance(pval, list):
                    loc, state = allocate(state, pval)
                    new_env = bind(new_env, pname, loc)
                else:
                    new_env = bind(new_env, pname, pval)
            return evaluate_expr(closure.body, new_env, state)

        case Var(name=name):
            dval = lookup(env, name)
            if isinstance(dval, Loc):
                return access(state,dval)
            if isinstance(dval, Closure):
                return dval                   
            if isinstance(dval, (int, bool, list)):
                return dval
            raise ValueError(f"Var '{name}' non lega né location, né funzione, né valore primitivo.")

        case Lambda(params=ps, body=b):
            return Closure(params=ps, body=b, env=env)
        case Bool(value=v):
            return v
        case Number(value=n):
            return n
        case Apply(op, args):
            arg_vals = [evaluate_expr(a, env, state) for a in args]

            dval = lookup(env, op)

            if isinstance(dval, Operator):
                expected_types, return_type = dval.type
                if len(expected_types) != len(arg_vals):
                    raise ValueError(
                        f"Operatore '{op}' si aspetta {len(expected_types)} argomenti, "
                        f"ne ha ricevuti {len(arg_vals)}"
                    )
                for i, (exp_t, actual) in enumerate(zip(expected_types, arg_vals), start=1):
                    if type(actual) is not exp_t:
                        raise ValueError(
                            f"Operatore '{op}', argomento {i} atteso {exp_t.__name__}, "
                            f"ottenuto {type(actual).__name__}"
                        )
                return dval.fn(arg_vals)

            if isinstance(dval, Closure):
                closure = dval
                if len(closure.params) != len(arg_vals):
                    raise ValueError(
                        f"Funzione '{op}' si aspetta {len(closure.params)} argomenti, "
                        f"ne sono stati forniti {len(arg_vals)}."
                    )
                new_env = closure.env
                for pname, pval in zip(closure.params, arg_vals):
                    if isinstance(pval, list):
                        loc, state = allocate(state, pval)
                        new_env = bind(new_env, pname, loc)
                    else:
                        new_env = bind(new_env, pname, pval)
                return evaluate_expr(closure.body, new_env, state)

            raise ValueError(f"'{op}' non è né un operatore né una funzione.")
        
        case Let(name, expression=expr, body=body):
            val = evaluate_expr(expr, env, state)
            extended_env = bind(env, name, val)
            return evaluate_expr(body, extended_env, state)

        case _:
            raise ValueError(f"Espressione non riconosciuta: {node}")

        
def execute_command(cmd: Command, env: Environment, state: State) -> tuple[Environment, State]:
    match cmd:
        case VarDecl(name=nm, expression=expr):
            val = evaluate_expr(expr, env, state)
            if isinstance(val, list):
                loc, state = allocate(state, val)
                env = bind(env, nm, loc)
                kind = "permutazione"
                print(f"\033[92mVariabile '{nm}' di tipo '{kind}' dichiarata.\033[0m")
                USER_VARS.append(nm)
            elif isinstance(val, bool):
                loc, state = allocate(state, val)
                env = bind(env, nm, loc)
                kind = "booleano"
                print(f"\033[92mVariabile '{nm}' di tipo '{kind}' dichiarata.\033[0m")
                USER_VARS.append(nm)
            elif isinstance(val, int):
                loc, state = allocate(state, val)
                env = bind(env, nm, loc)
                kind = "intero"
                print(f"\033[92mVariabile '{nm}' di tipo '{kind}' dichiarata.\033[0m")
                USER_VARS.append(nm)
            elif isinstance(val, Closure):
                env = bind(env, nm, val)
                print(f"\033[92mFunzione '{nm}' dichiarata.\033[0m")
                USER_VARS.append(nm)
            else:
                raise ValueError(f"Dichiarazione non valida per '{nm}'.")
            return env, state

        case Assign(name=nm, expression=expr):
            dval = lookup(env, nm)
            if not isinstance(dval, Loc):
                raise ValueError(f"Non puoi assegnare a '{nm}' perché non è variabile.")
            val = evaluate_expr(expr, env, state)
            if isinstance(val, list):
                state = update(state, dval, val)
                kind = "permutazione"
            elif isinstance(val, int):
                state = update(state, dval, val)
                kind = "intero"
            elif isinstance(val, bool):
                state = update(state, dval, val)
                kind = "booleano"
            else:
                raise ValueError(
                    f"Assegnamento a '{nm}' richiede permutazione, intero o booleano, "
                    f"ma ho ottenuto: {val!r}"
                )
            print(f"\033[92mVariabile '{nm}' aggiornata di tipo '{kind}'.\033[0m")
            return env, state

        case Print(expression=expr):
            dval = evaluate_expr(expr, env, state)
            if isinstance(dval, list):
                print(f"\033[93mScrittura in cicli:\033[0m \033[96m{scrittura_in_cicli(dval)}\033[0m")
                print(f"\033[93mScrittura come lista:\033[0m \033[96m{dval}\033[0m")
            elif isinstance(dval, bool):
                print(f"\033[93mil valore di verità è:\033[0m \033[96m{dval}\033[0m")
            elif isinstance(dval, int):
                print(f"\033[93mIntero corrispondente:\033[0m \033[96m{dval}\033[0m")
            return env, state
        
        case EvalCommand(expression=FunctionApp(name="productVisualize", args=args)):
            if len(args) < 2:
                raise ValueError("productVisualize richiede almeno due permutazioni.")
            perms: List[List[int]] = []
            names: List[str] = []
            for arg in args:
                if not isinstance(arg, Var):
                    raise ValueError("productVisualize richiede variabili come argomenti.")
                dval = lookup(env, arg.name)
                if not isinstance(dval, Loc):
                    raise ValueError(f"'{arg.name}' non è una variabile definita.")
                perm = access(state,dval)
                perms.append(perm.copy())
                names.append(arg.name)
            current = perms[0]
            current_name = names[0]
            for next_perm, next_name in zip(perms[1:], names[1:]):
                composed = prodotto_permutazioni(current, next_perm)
                anima_treccia_morph(current.copy(),
                                   composed.copy(),
                                   f"{current_name}*{next_name}",
                                   current_name)
                current = composed
                current_name = f"{current_name}*{next_name}"
            print(f"\033[93mScrittura in cicli:\033[0m \033[96m{scrittura_in_cicli(current)}\033[0m")
            print(f"\033[93mScrittura come lista:\033[0m \033[96m{current}\033[0m")
            return env, state
        
        case EvalCommand(expression=expr):
            val = evaluate_expr(expr, env, state)
            if isinstance(val, list):
                print(f"\033[93mScrittura in cicli:\033[0m \033[96m{scrittura_in_cicli(val)}\033[0m")
                print(f"\033[93mScrittura come lista:\033[0m \033[96m{val}\033[0m")
            elif isinstance(val, bool):
                print(f"\033[93mValore booleano:\033[0m \033[96m{val}\033[0m")
            elif isinstance(val, int):
                print(f"\033[93mIntero:\033[0m \033[96m{val}\033[0m")
            return env, state
        
        case IfElse(cond, then_branch, else_branch):
            cond_val = evaluate_expr(cond, env, state)
            if not isinstance(cond_val, bool):
                raise ValueError("If condition must be boolean")
            saved_next_loc = state.next_loc
            if cond_val:
                then_env = env
                _, state1 = execute_command_seq(then_branch, then_env, state)
                state2 = State(store=state1.store, next_loc=saved_next_loc)
                return env, state2
            else:
                else_env = env
                _, state1 = execute_command_seq(else_branch, else_env, state)
                state2 = State(store=state1.store, next_loc=saved_next_loc)
                return env, state2

        case While(cond, body):
            cond_val = evaluate_expr(cond, env, state)
            if not isinstance(cond_val, bool):
                raise ValueError("While condition must be boolean")
            saved_next_loc = state.next_loc
            if cond_val:
                body_env = env
                _, state1 = execute_command_seq(body, body_env, state)
                state2 = State(store=state1.store, next_loc=saved_next_loc)
                return execute_command(While(cond, body), env, state2)
            else:
                return env, state

        case BraidDiagram(name=nm):
            dval = lookup(env, nm)
            if not isinstance(dval, Loc):
                raise ValueError(f"braidDiagram su '{nm}' non è variabile.")
            perm = access(state,dval)
            mostra_diagramma(perm, nm)
            return env, state

        case CyclesDiagram(name=nm):
            dval = lookup(env, nm)
            if not isinstance(dval, Loc):
                raise ValueError(f"cyclesDiagram su '{nm}' non è variabile.")
            perm = access(state,dval)
            mostra_cycles_diagramma(perm, nm)
            return env, state

        case ListCmd():
            for name in sorted(USER_VARS):
                val = lookup(env, name)
                if isinstance(val, Loc):
                    v = access(state, val)
                else:
                    v = val
                print(f"{name} = {v}")
            return env, state

        case Help(topic=t):
            DESCRIZIONI = {
                "clear":       "\033[92mSINTASSI: clear\033[0m\n\033[93mElimina tutte le variabili utente e ripristina l'ambiente allo stato iniziale.\033[0m",
                "list":        "\033[92mSINTASSI: list\033[0m\n\033[93mMostra le variabili attualmente definite e i loro valori.\033[0m",
                "print":       "\033[92mSINTASSI: print(permutazione)\033[0m\n\033[93mStampa la permutazione sia come lista di interi sia in forma algebrica.\033[0m",
                "let": "\033[92mSINTASSI: let nome = expr1 in expr2\033[0m\n\033[93mValuta expr2 nell’ambiente esteso con nome legato al valore temporaneo expr1.\033[0m",
                "if":          "\033[92mSINTASSI: if cond then ... else ... endif\033[0m\n\033[93mValuta cond e, se vero, esegue il blocco then, altrimenti il blocco else.\033[0m",
                "while":       "\033[92mSINTASSI: while cond do ... done\033[0m\n\033[93mFinché cond è vero, ripete il corpo del loop.\033[0m",
                "lambda":      "\033[92mSINTASSI: lambda(param1, ..., paramN) expr\033[0m\n\033[93mCrea una funzione anonima con i parametri dati.\033[0m",
                "braidDiagram":    "\033[92mSINTASSI: braidDiagram(nomeVariabile)\033[0m\n\033[93mMostra il diagramma a treccia della permutazione.\033[0m",
                "cyclesDiagram":   "\033[92mSINTASSI: cyclesDiagram(nomeVariabile)\033[0m\n\033[93mMostra il diagramma a cicli della permutazione.\033[0m",
                "cyclesNumber":    "\033[92mSINTASSI: cyclesNumber(permutazione)\033[0m\n\033[93mStampa il numero di cicli della permutazione.\033[0m",
                "fix":             "\033[92mSINTASSI: fix(permutazione)\033[0m\n\033[93mStampa i punti fissi della permutazione.\033[0m",
                "identity":        "\033[92mSINTASSI: identity(n)\033[0m\n\033[93mGenera la permutazione identità su n elementi.\033[0m",
                "inverse":         "\033[92mSINTASSI: inverse(permutazione)\033[0m\n\033[93mCalcola l'inversa della permutazione.\033[0m",
                "power":           "\033[92mSINTASSI: power(permutazione, esponente)\033[0m\n\033[93mEleva la permutazione all'esponente intero non negativo.\033[0m",
                "conjugate":       "\033[92mSINTASSI: conjugate(q, s)\033[0m\n\033[93mCalcola q · s · q⁻¹.\033[0m",
                "order":           "\033[92mSINTASSI: order(permutazione)\033[0m\n\033[93mStampa l'ordine della permutazione (mcm delle lunghezze dei cicli).\033[0m",
                "parity":          "\033[92mSINTASSI: parity(permutazione)\033[0m\n\033[93mStampa se la permutazione è pari o dispari.\033[0m",
                "shift":           "\033[92mSINTASSI: shift(permutazione, offset)\033[0m\n\033[93mTrasla tutti gli elementi della permutazione di offset pos./neg.\033[0m",
                "productVisualize":"\033[92mSINTASSI: productVisualize(perm1, perm2, ..., permk)\033[0m\n\033[93m restituisce la permutazione prodotto perm1*perm2*...*permk e mostra un'animazione per ogni prodotto *\033[0m",
                "help":            "\033[92mSINTASSI: help [comando]\033[0m\n\033[93mSenza argomenti mostra la lista dei comandi; con argomento, i dettagli di quel comando.\033[0m"
            }
            if t is None:
                comandi = sorted(DESCRIZIONI.keys())
                col1 = comandi[::2]
                col2 = comandi[1::2] + [""]
                print("\033[93mComandi disponibili:\033[0m\n")
                for a, b in zip(col1, col2):
                    print(f"  \033[93m{a:<15}\033[0m \033[93m{b:<15}\033[0m")
            else:
                desc = DESCRIZIONI.get(t)
                if desc:
                    print(desc)
                else:
                    print(f"\033[91mComando sconosciuto: {t}\033[0m")
            return env, state

        case Clear():
            env = init_env()
            USER_VARS.clear()
            state = empty_state()
            print("\033[92mAmbiente ripristinato.\033[0m")
            return env, state

        case _:
            raise ValueError(f"Comando non riconosciuto: {cmd}")

def execute_command_seq(seq: CommandSequence, env: Environment, state: State) -> tuple[Environment, State]:
    env, state = execute_command(seq.first, env, state)
    if seq.rest:
        return execute_command_seq(seq.rest, env, state)
    return env, state

"""----------------------------------
            LOOP PRINCIPALE
----------------------------------"""

def init_env() -> Environment:
    env = empty_environment()
    env = bind(env, "+", Operator(type=([int, int], int), fn=lambda args: args[0] + args[1]))
    env = bind(env, "-", Operator(type=([int, int], int), fn=lambda args: args[0] - args[1]))
    env = bind(env, "*", Operator(type=([int, int], int), fn=lambda args: args[0] * args[1]))
    env = bind(env, "/", Operator(type=([int, int], int), fn=lambda args: args[0] // args[1]))
    env = bind(env, "%", Operator(type=([int, int], int), fn=lambda args: args[0] % args[1]))
    env = bind(env, "==", Operator(type=([int, int], bool), fn=lambda args: args[0] == args[1]))
    env = bind(env, "!=", Operator(type=([int, int], bool), fn=lambda args: args[0] != args[1]))
    env = bind(env, "<",  Operator(type=([int, int], bool), fn=lambda args: args[0] <  args[1]))
    env = bind(env, ">",  Operator(type=([int, int], bool), fn=lambda args: args[0] >  args[1]))
    env = bind(env, "<=", Operator(type=([int, int], bool), fn=lambda args: args[0] <= args[1]))
    env = bind(env, ">=", Operator(type=([int, int], bool), fn=lambda args: args[0] >= args[1]))
    env = bind(env, "and", Operator(type=([bool, bool], bool), fn=lambda args: args[0] and args[1]))
    env = bind(env, "or",  Operator(type=([bool, bool], bool), fn=lambda args: args[0] or args[1]))
    env = bind(env, "not", Operator(type=([bool], bool), fn=lambda args: not args[0]))
    env = bind(env, "env", Closure(params=[], body=Var("env"), env=env))
    
    for name, params, make_name in permutation_builtins:
        env = bind(env, name, Closure(
            params=params,
            body=FunctionApp(
                name=make_name,
                args=[Var(p) for p in params] + [Var("env")]
            ),
            env=env
        ))
    return env

def REPL():
    env = init_env()
    state = empty_state()
    
    global BUILTIN_NAMES
    BUILTIN_NAMES = set(
        ["+", "-", "*", "/", "%", "==", "!=", "<", ">", "<=", ">=", "and", "or", "not", "env"]
        + [name for name, _, _ in permutation_builtins]
    )

    print(f"\033[92mInterprete avviato.\033[0m")
    print(f"\033[93mDigita \033[94m'exit'\033[93m per uscire.")
    print(f"Digita \033[94m'help'\033[93m per la lista dei comandi,")
    print(f"oppure \033[94m'help nomeComando'\033[93m per i dettagli di un comando.\033[0m\n")

    while True:
        try:
            line = input("> ").strip()
            if not line:
                continue
            if line.lower() == "exit":
                print("\033[92mUscita...\033[0m")
                break

            seq_ast = parse_program(line)
            env, state = execute_command_seq(seq_ast, env, state)

        except LarkError as e:
            print(f"\033[91mErrore di sintassi: {e}\033[0m\n")
        except ValueError as ve:
            print(f"\033[91mErrore semantico: {ve}\033[0m\n")
        except TypeError as te:
            print(f"\033[91mErrore di tipo: {ex}\033[0m\n")
        except Exception as ex:
            print(f"\033[91mErrore imprevisto: {ex}\033[0m\n")

if __name__ == "__main__":
    REPL()

