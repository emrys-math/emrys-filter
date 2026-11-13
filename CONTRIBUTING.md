# Contributing

EmrysFilter implements algorithms from the public literature and contributes none of its own
([docs/design/decisions.md](docs/design/decisions.md), [published-algorithms-only]). Every algorithm
in the headers cites its source in [docs/reference/bibliography.md](docs/reference/bibliography.md),
says what the code takes from it, and says where it departs. That rule shapes how features are
proposed and how open problems are handled. It also obliges the library to leave room: a method it
won't take must be writable against its public contracts, without a fork ([user-extension-points]).

## Proposing a feature

Open a **Feature request** issue. The form asks for the need and for the literature: the paper,
book, or thesis the feature would implement, cited by first author, year, title, and venue, with
the section or equation numbers the code would follow. A preprint is fine. A method that has no
reference yet can't go into the library, but it can go on the research page below.

Before opening one, read [docs/design/roadmap.md](docs/design/roadmap.md) for what is planned
and its order, [docs/design/decisions.md](docs/design/decisions.md) for the decisions already
made, and the open questions in the design reviews in `docs/guide/`. A request that fits a planned step is still
worth filing: it adds the reference and the use case to the step.

## Open problems for researchers

[docs/research.md](docs/research.md) lists needs the library has that no published method fills
yet, or fills only without the guarantee the library would want: a bound, an error analysis, or
a proof that an approximation is what it claims. Each is scoped so that a graduate student can
prototype it against the library's hooks, prove something about it, and publish it. Once it is
published, the library implements it and cites it. To take one, open a **Research project**
issue so that nobody else starts the same one.

## Code

Build and test with `cmake --workflow --preset clang` (or `gcc`, or `msvc` from an x64 Visual
Studio developer environment, since every preset uses Ninja); CI runs all three,
clang-format, a header-layer check, and a Doxygen build that fails on any warning, when a
maintainer dispatches it, so run them locally before opening a pull request. The
conventions the code follows are in [docs/design/style.md](docs/design/style.md): names,
parameter order, allocation, `std::expected` errors, no exceptions, and the citation convention.
[AGENTS.md](AGENTS.md) holds the workflow, for people and coding agents alike. A change to the
architecture changes `docs/design/decisions.md` in the same commit, and a step's progress is
recorded in `docs/design/roadmap.md`.
