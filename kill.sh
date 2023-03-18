#!/bin/bash

A="$1"


if [ $A = "va" ]
then
    echo "kill va"
    pid=$(ps ax | grep $A | grep -v grep | awk '{ print $1 }')
    echo $pid
elif [ $A = "expr" ]
then
    echo "kill expr"
elif [ $A = "au" ]
then
    echo "kill au"
    local pid
    pid=$(ps ax | grep $A | grep -v grep | awk '{ print $1 }')
    echo $pid
fi
