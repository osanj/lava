#!/bin/bash
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
pytest -v --junitxml ${dir}/results.xml ${dir}/../test
