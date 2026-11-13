# EmrysFilter

Header-only C++23 state estimation library on Eigen 5.0, by Emrys Applied Mathematics. Headers live
in `include/EmrysFilter/` in the namespace `Emrys`, in six layered directories [layered-headers]:
`Core/`, `Spaces/`, `Distributions/`, `Models/`, `Filters/`, and `Smoothers/`, each with an umbrella header beside
it (`EmrysFilter/Spaces.h`), and `EmrysFilter.h` includes them all. A header includes another as
`<EmrysFilter/Spaces/Space.h>` and never from a later layer; `tools/check_layers.py` checks that, in
CI too. GoogleTest tests live in `tests/`, one file per area, in the directory that mirrors the
header's. The library was eigen-filter until roadmap step 13, and its repository is
`emrys-math/emrys-filter`.

## Which documents to read

- **Using the library** in a project of your own: [docs/usage.md](docs/usage.md), the programs
  in [examples/](examples/) that it points to, and [docs/choosing.md](docs/choosing.md), the
  reasoning behind its choice of filter. [skills/](skills/) holds the skills a
  user's agent installs; usage.md says how. The rest of this file is for working on the library.
- **Working on the library:** [docs/design/style.md](docs/design/style.md) before editing
  anything under `include/`, `tests/`, `examples/`, or `benchmarks/`; the decisions in
  [docs/design/decisions.md](docs/design/decisions.md) that touch your change; and
  [docs/design/roadmap.md](docs/design/roadmap.md) for the ordered steps and their status.
  [docs/design/parallelism.md](docs/design/parallelism.md) is the plan behind steps 18 and 19
  (execution policies and GPU execution), not yet decided.
  [docs/design/benchmarks.md](docs/design/benchmarks.md) is the plan for the benchmarks and the
  rule that separates them from the tests: a test checks the implementation, a benchmark backs
  a claim.

## Before starting work

- The library is pre-1.0, so a breaking change is still allowed and cheaper now than later:
  never defer a rename, a signature change, or any other change that could break code. Only
  additions that break nothing may wait. Every change a user can see goes into `CHANGELOG.md`
  under Unreleased in the commit that makes it, and a breaking one says what to write instead.
- Treat recorded decisions as settled. If one turns out to be wrong, change it in `decisions.md`
  in the same commit as the code, and give the reason in the commit message.
- `roadmap.md` is the handoff between sessions, and changes in the commit with the code: a
  finished item is deleted, a finished step becomes a row of its Done table, and a paused step
  keeps only what the next session needs to continue. History goes in the commit message.
- Design reviews of each area live in `docs/guide/`. [docs/guide/README.md](docs/guide/README.md)
  lists them, their scope, and how to write one. An open design question goes in the review for
  its area, as a numbered question.
- `dev` is the public branch, and its history is never rewritten: no rebase, amend, or force-push
  of a commit already on it. `next` reaches `dev` by fast-forward only, and a mistake on `dev` is
  fixed by a new commit. Only the user decides to rewrite `dev`.

## Build and test

```sh
cmake --workflow --preset clang   # or gcc, msvc
```

Every preset uses Ninja, so `msvc` runs from an x64 developer environment. MSVC and GCC/Clang
disagree on heavy template code, so check both families before calling template-heavy work done.
Run the compiler workflows one after another, never in parallel. A test translation unit takes
one to four gigabytes, so Ninja compiles at most `EMRYS_FILTER_TEST_COMPILE_JOBS` (12) of them at
once, and GCC and Clang Debug test builds use `-Og` with line tables only; configure with
`-DEMRYS_FILTER_TEST_FULL_DEBUG_INFO=ON` to step through a test in a debugger.

Checks run in stages [check-stages], and `python tools/required_checks.py land` (or `pr`) prints
the ones a change needs, with their commands:

- **While working**, build what you are changing with the `clang` preset and run
  `ctest --preset clang -E example.consumer`, which leaves out the slow `example.consumer`.
- **`land`**, before work goes onto `next`: the `clang` workflow, which leaves out the negative
  compile tests, and the checks that take seconds: clang-format, `tools/check_layers.py`,
  `tools/check_decisions.py`, and `tools/check_compile_errors.py`. MSVC joins when a deduction
  guide or execution policy changed, and the `clang-compile-errors` workflow when a claimed compile
  error or its case changed.
- **`pr`**, before `next` goes to `dev`: GCC, MSVC, the negative compile tests with each of Clang,
  GCC, and MSVC (the `<compiler>-compile-errors` workflows, which compile again only the cases
  whose file's object was rebuilt), the Doxygen build and the policy check over its XML,
  `clang-sanitize` in WSL, and coverage, one at a time, and the CI jobs the script names,
  dispatched when the user agrees.
- **`ci`**, before a release: every CI job.

A change that needs no compiling, such as documentation or instructions, needs only the checks the
script names, and can be committed without a build and rebased afterward. Otherwise don't build or
test before a rebase onto a branch that already exists: rebase first, then build once. The test
presets run eight tests at once, and a negative compile test counts as two, so a test run compiles
at most four cases at once.

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs the same three workflow presets,
GCC 14 and Clang 21 on Linux and MSVC on Windows, each followed by its `-compile-errors` workflow; the `clang-sanitize` preset (Clang 21 with
AddressSanitizer and UndefinedBehaviorSanitizer), `gcc-no-exceptions` (`-fno-exceptions
-fno-rtti`), and `clang-libcxx` (LLVM's library), all on Linux; GCC on Linux arm64; the
`apple-clang` preset on macOS; the Cortex-M build in `benchmarks/cortex_m/` under QEMU
[embedded-builds]; the coverage of the headers by the tests, a `clang-coverage` build measured by
`tools/coverage.py` [coverage];
the vcpkg port in `packaging/vcpkg/`, installed from the checkout by `tools/vcpkg_test_port.py` and
used by `examples/consumer`; the Conan recipe in `packaging/conan/`, created from the checkout by
`tools/conan_test_recipe.py` with GCC 14 and Clang 21; a
clang-format check, a header-layer check, a check that each claimed compile error has a negative
test (`tools/check_compile_errors.py`), and a Doxygen build that fails on any warning, followed by a
check over its XML that each policy namespace holds only its kind (`tools/check_policies.py`). The tests, examples, and benchmarks build with warnings as errors
(`EMRYS_FILTER_WARNINGS_AS_ERRORS`). CI runs only when dispatched by hand, and only the jobs
ticked, since a full run takes hours to finish: nothing runs on a push or a pull request, so
run the local checks above, and dispatch a job only when the user asks for it
(`gh workflow run CI --ref next -f gcc=true -f docs=true`). The Linux jobs but `clang-libcxx`
install TBB, so libstdc++ runs `std::execution::par` on threads; without TBB it runs on one
thread. The examples build with the tests and run as ctest
tests named `example.<name>`, each failing when its filter's consistency check fails;
`example.consumer` builds `examples/consumer/`, the project a user copies, against the checkout
fetched and then installed [cmake-package];
and the test `skills.install` installs the user skills and checks them [user-skills].

The reference documentation (roadmap step 13b) is a Doxygen build of every header, internals
included, for the maintainer; `-DEMRYS_FILTER_DOCS_INTERNALS=OFF` leaves out the `internal`
namespace and private members, which is the reference that the `Pages` workflow
([.github/workflows/pages.yml](.github/workflows/pages.yml)) publishes to
https://emrys-math.github.io/emrys-filter/ when dispatched (`gh workflow run Pages --ref dev`). Both
fail on any warning. It needs Doxygen on PATH (`winget install DimitriVanHeesch.Doxygen`
on Windows) or `-DDOXYGEN_EXECUTABLE=...`:

```sh
cmake -S . -B build/docs -DEMRYS_FILTER_BUILD_DOCS=ON -DEMRYS_FILTER_BUILD_TESTS=OFF
cmake --build build/docs --target docs   # build/docs/docs/html, warnings in build/docs/docs/warnings.txt
```

The module pages are the place to start reading an area.
