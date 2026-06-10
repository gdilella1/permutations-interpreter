# Finite Permutation Interpreter (Custom DSL)

A mathematically rigorous Domain-Specific Language (DSL) and interpreter designed for the analysis, transformation, and visualization of finite permutations expressed in cyclic algebraic notation. This tool serves as a computational engine for symmetric group structures ($S_n$) and an educational companion for abstract algebra and computational mathematics.

## 🚀 Key Features

* **Algebraic Cyclic Input:** Direct parsing of finite permutations with explicit tracking of fixed points to preserve the exact cardinality of the acting set.
* **Advanced Language Constructs:** Supports variables (permutations, integers, booleans), conditional blocks (`if-then-else`), loops (`while-do`), and first-class anonymous functions (`lambda` abstractions)[cite: 1].
* **Functional Scoping:** Lexical block-local static scoping via immutable environments and state allocation[cite: 1].
* **Built-in Algebraic Operators:** Runtime execution of products (composition), inverses, powers, conjugates, order calculations (via LCM of cycle lengths), parity checks, shifts, and fixed-point extractions[cite: 1].
* **Graph & Topological Visualizations:** Dynamic and static visualization generation, including **Braid Diagrams** (via Matplotlib) and **Cycle Diagrams** (via NetworkX)[cite: 1].
* **Algorithmic Animations:** Includes `productVisualize`, which animates the composition of $k$-permutations sequentially using morphing braid tracking algorithms[cite: 1].

## 🛠️ Architecture & Design

### Parsing & AST Transformation
The syntactic analysis is powered by the **Lark** parsing library[cite: 1]. The parser processes the custom grammar rules into an intermediate concrete parse tree, which is then recursively transformed via localized functions (`transform_expression`, `transform_command_tree`) into a deeply typed Abstract Syntax Tree (AST)[cite: 1].

### Memory & Execution Model
* **Environment:** An immutable mapping from identifiers to denotable values (`DVal`), such as memory locations or function closures[cite: 1].
* **State & Store:** A centralized store mapping integer addresses (`Loc`) to mutable values (`MVal`)[cite: 1].
* **Lexical Blocks:** Costructs like `let In` expressions utilize short-lived environments that clear local bindings upon evaluation, guaranteeing zero memory leaks and strict encapsulation[cite: 1].

## 📁 Repository Structure

Following professional software engineering standards, the repository isolates the execution runtime from the documentation assets[cite: 1]:

```text
├── src/
│   └── interpreter.py     # The complete, unified Python engine (Parser & Evaluator)
└── docs/
    └── documentation.pdf  # Full academic and technical documentation report
