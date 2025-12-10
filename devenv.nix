{ pkgs, ... }:

{
  # https://devenv.sh/basics/
  env.GREET = "devenv for Rust";

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
  ];

  # https://devenv.sh/languages/
  languages.rust.enable = true;

  # https://devenv.sh/processes/
  # processes.cargo-watch.exec = "cargo-watch";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  enterShell = ''
    echo "🦀 Rust stable development environment loaded"
    echo "Run 'hello' to test the environment"
    cargo --version
    rustc --version
  '';

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    cargo test
  '';

  # https://devenv.sh/pre-commit-hooks/
  # pre-commit.hooks.shellcheck.enable = true;

  # Make diffs fantastic
  # difftastic.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
