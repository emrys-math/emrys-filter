# Security policy

## Reporting a vulnerability

Report a vulnerability privately, through
[GitHub's private reporting](https://github.com/emrys-math/emrys-filter/security/advisories/new)
(the repository's **Security** tab, **Report a vulnerability**), and not in a public issue. Give
the release or commit, the compiler and its flags (whether `NDEBUG` was defined), the Eigen
version, and a program that shows the behavior.

The library has one maintainer, who answers reports as time allows; there is no promised response
time. A confirmed vulnerability is fixed on `dev`, released, and published as a GitHub security
advisory that credits the reporter, unless the reporter asks not to be named.

## Supported versions

Before 1.0, only the latest release is supported. A fix goes into the next release, and none is
back-ported to an earlier one.

## What counts as a vulnerability

EmrysFilter is a header-only library. It opens no files or sockets, parses no input, and holds no
credentials. It runs inside the program that includes it, on numbers the program passes in, and
those numbers may come from sensors or from messages the program does not control. A
vulnerability is a way for those numbers, or for something the project ships, to do harm beyond
producing a wrong estimate.

Report privately:

- Undefined behavior, such as an out-of-bounds access or a NaN converted to an integer, that a
  floating-point value passed to the library reaches, whether NaN, infinite, subnormal, or finite.
  This counts in a build with `NDEBUG` even where an `EMRYS_FILTER_ASSERT` checks the value in a
  Debug build: the library's code must stay defined when that assertion is compiled out
  ([decisions.md](docs/design/decisions.md), [value-preconditions]).
- A call that never returns, or that allocates more than the sizes the program chose, because of
  such a value.
- Undefined behavior, including a data race under an execution policy, that a call meeting every
  documented precondition reaches.
- A crash on an input for which the documentation promises a `std::expected` error.
- A build or install that fetches, runs, or writes something other than what it declares: the
  CMake build, which fetches Eigen 5.0.1 checked against its SHA-256 when it finds no Eigen; the
  vcpkg port in `packaging/vcpkg/`; and the Conan recipe in `packaging/conan/`.
- Text in an agent skill under `skills/`, installed by `emrys_filter_add_skills_target`, that
  would lead a user's coding agent to run a command, fetch content, or send data for any purpose
  other than building and checking the user's estimator. A coding agent with shell access follows
  a skill as instructions, so the skills are held to the same standard as the code.

Report as a public issue, with the bug report form:

- A wrong or diverging estimate, a NaN state that follows from a NaN input, or a failed
  consistency check. The library promises that its behavior stays defined on any value, not that
  its estimate is right on any value.
- A failed `EMRYS_FILTER_ASSERT` on a size, an index, or a type. These are preconditions, and the
  calling program is responsible for meeting them, as it is with Eigen's.

Not in scope:

- Defects in Eigen, the compiler, or the standard library; report them to their projects.
- Behavior of code the program supplies to the library: its models, sensors, policies, and
  allocators.
- Code that runs only in the library's own development: the tests, `benchmarks/`, `tools/`, and
  the CI workflows, which run only when a maintainer dispatches them.
- Predictable random draws. The library draws from the generator the program passes in, through
  the standard distributions, and nothing it computes is secret; its sampling is not for making
  anything that must be.
- Timing differences. The library makes no constant-time claim.
