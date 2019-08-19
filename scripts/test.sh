#!/bin/bash
dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"
pytest -v \
    --junitxml ${dir}/test_results.xml \
    --cov ${dir}/../lava \
    --cov-report xml:${dir}/test_coverage.xml \
    ${dir}/../test
