# Contributing

That would be awesome if you want to contribute something to TileLang!

- [Contributing](CONTRIBUTING.md#contributing)
  - [Open an Issue First](CONTRIBUTING.md#open-an-issue-first)
  - [Reporting Bugs](CONTRIBUTING.md#reporting-bugs)
  - [Asking Questions](CONTRIBUTING.md#asking-questions)
  - [Submitting Pull Requests](CONTRIBUTING.md#submitting-pull-requests)
  - [Repository Setup](CONTRIBUTING.md#repository-setup)
  - [Running Tests](CONTRIBUTING.md#running-tests)

## Open an Issue First

**Please open an issue before submitting a pull request.** This allows us to discuss the problem or feature, agree on an approach, and avoid duplicated or unnecessary work. PRs without a linked issue may not be reviewed.

You can use the provided issue templates to file:

- 🐛 [Bug Reports](https://github.com/platelett/tilelang-ascend/issues/new?template=bug-report.yml)
- ✨ [Feature Requests](https://github.com/platelett/tilelang-ascend/issues/new?template=feature-request.yml)
- 🤔 [Questions](https://github.com/platelett/tilelang-ascend/issues/new?template=questions.yml)

## Reporting Bugs

If you run into any weird behavior while using TileLang, feel free to open a new issue in this repository! Please run a **search before opening** a new issue, to make sure that someone else hasn't already reported or solved the bug you've found.

Any issue you open must include:

- Code snippet that reproduces the bug with a minimal setup.
- A clear explanation of what the issue is.


## Asking Questions

Please ask questions in issues.

## Submitting Pull Requests

**Before opening a PR, please make sure there is an open issue that describes the problem or feature.** Reference the issue in your PR description (e.g., `Closes #123`).

Please run `./format.sh` before submitting a pull request to make sure that your code is formatted correctly.

Please include tests and docs with every pull request!

## Repository Setup

To run the build, you need to have the TileLang repository cloned to your computer. After that, you need to `cd` into the directory where you cloned it, and install the dependencies with `python`:

```bash
python setup.py install
```


## Running Tests

To run the tests, start by building the project as described in the [Repository Setup](CONTRIBUTING.md#repository-setup) section.

Then you can rerun the tests with:

```text
python -m pytest testing
```

