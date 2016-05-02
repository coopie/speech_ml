#!/bin/sh


nosetests --tests=`find src -name "*_test.py" | tr "\n" ", " | sed "s/,$//g"`
