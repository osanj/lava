#!/bin/bash

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"


latest_tag=$(git tag -l --points-at HEAD 2>&1)
version_in_code=$(grep '__version__' ${dir}/../lava/__init__.py | grep -oP '([0-9]+\.[0-9]+\.[0-9]+[a-zA-Z0-9-._]*)')

# check versions
if [ "$latest_tag" != "$version_in_code" ]; then
    echo "git tag \"$latest_tag\" does not match with version in source code \"$version_in_code\""
    exit 1
fi

echo "Releasing version $version_in_code..."


# check for untracked / uncommitted files
count_staged=$(cd ${dir}/.. && git diff --name-only | grep "lava/" | wc -l)
count_not_staged=$(cd ${dir}/.. && git diff --name-only --cached | grep "lava/" | wc -l)
count_not_added=$(cd ${dir}/.. && git ls-files --other --directory --exclude-standard | grep "lava/" | wc -l)

if [ "$count_staged" -gt "0" ] || [ "$count_not_staged" -gt "0" ] || [ "$count_not_added" -gt "0" ]; then
    echo "the git workspace is dirty, there exist not committed files"
    exit 1
fi

echo "Workspace is clean..."


# build distribution
rm -r ${dir}/../dist
cd ${dir}/.. && python3 setup.py sdist bdist_wheel
echo "Built wheel..."


# publish on pypi
if [ "$1" = "--release" ]; then
    python3 -m twine upload dist/*
else
    python3 -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
fi
