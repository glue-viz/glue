#!/bin/sh

if ! [ $TRAVIS_REPO_SLUG ]; then
    echo "Not building on travis. Exiting"
    exit 0
fi

if [ $TRAVIS_REPO_SLUG != "glue-viz/glue" ]; then
   echo "Not building from main repo. Exiting"
   exit 0
fi

travis login --github-token=$GITHUB_TOKEN
echo "Travis MacGlue Branch Summary"
travis branches -r glue-viz/Travis-MacGlue

job_id=`travis branches -r glue-viz/Travis-MacGlue | grep $TRAVIS_BRANCH | cut -d" " -f 2-3 | cut -d '#' -f2`

echo "job_id is $job_id"

if ! [ $job_id ]; then
   echo "Could not find a Travis-MacGlue branch named $TRAVIS_BRANCH. Exiting"
   exit 0
fi

travis restart $job_id -r glue-viz/Travis-MacGlue
