# Contributing

EmrysFilter implements algorithms from the public literature and contributes none of its own
([docs/design/decisions.md](docs/design/decisions.md), [published-algorithms-only]). Every algorithm
in the headers cites its source in [docs/reference/bibliography.md](docs/reference/bibliography.md),
says what the code takes from it, and says where it departs. That rule shapes how features are
proposed and how open problems are handled. It also obliges the library to leave room: a method it
won't take must be writable against its public contracts, without a fork ([user-extension-points]).

## License of contributions

The library is under the Apache License 2.0 with LLVM Exceptions ([LICENSE](LICENSE)). By its
section 5, a contribution you submit is under the same license unless you say otherwise when you
submit it, so no separate contributor agreement is needed.

## Conduct

Everyone who takes part in the project follows the [Code of Conduct](CODE_OF_CONDUCT.md). It says
which topics belong in the project's spaces, the rules that discussion follows, and how to report
a violation.

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

Every pull request that adds or changes behavior adds or changes the tests for it, in the same
commit. A new feature comes with tests that instantiate each public entity it adds and run each
of its code paths, and a bug fix comes with a test that fails without the fix. The rule in full is
in the Tests section of [docs/design/style.md](docs/design/style.md); a pull request without the
tests it needs is not merged.

Build and test with `cmake --workflow --preset clang` (or `gcc`, or `msvc` from an x64 Visual
Studio developer environment, since every preset uses Ninja); CI runs all three,
clang-format, a header-layer check, and a Doxygen build that fails on any warning, on each pull
request, choosing the jobs your change needs with `python tools/required_checks.py pr`. Run them
locally before opening a pull request, since the full suite takes CI hours. Before each release,
clang-tidy analyzes the headers with the checks in `.clang-tidy`
(`python tools/clang_tidy.py build/clang`), and a finding is fixed or suppressed at its line with
the reason. The
conventions the code follows are in [docs/design/style.md](docs/design/style.md): names,
parameter order, allocation, `std::expected` errors, no exceptions, and the citation convention.
[AGENTS.md](AGENTS.md) holds the workflow, for people and coding agents alike. A change to the
architecture changes `docs/design/decisions.md` in the same commit, and a step's progress is
recorded in `docs/design/roadmap.md`.

A test build is heavy. Measured with `tools/compile_cost.py --match` on 2026-11-22 (Debug, one
compile at a time), the 117 test, example, and benchmark sources take 25
CPU-minutes to compile with Clang 21 and 37 with MSVC 19.50. A test file's median peak memory is
1.5 GiB with Clang and 1.9 GiB with MSVC, and the largest, `tests/Filters/sqrt_filters.cpp`, takes
5.4 GiB and 49 seconds with Clang, 8.0 GiB and 103 seconds with MSVC. The presets compile at most
`EMRYS_FILTER_TEST_COMPILE_JOBS` (12) test files at once, which needs about 2 GiB a job on
average and more when the largest files compile together: their 12 largest come to 38 GiB with
Clang and 55 GiB with MSVC. With less than 32 GiB of memory, configure with a smaller value, about
one job per 2.5 GiB, for example `-DEMRYS_FILTER_TEST_COMPILE_JOBS=6` with 16 GiB; past that the
compilers swap or are killed. Under Clang, 64% of the time is the front end, most of it
template instantiation, and the rest code generation.
