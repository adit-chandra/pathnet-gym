#!/bin/bash

ps_num=10
for i in `eval echo {0..$ps_num}`
do
  python test.py &
done
