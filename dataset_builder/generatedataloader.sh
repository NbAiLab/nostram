#!/bin/bash
if test $# -ne 4
then
echo "Usage:$0 <templatefile><nosplits><name><outputdir>"
exit
fi
templatefile=$1
nosplits=$2
name=$3
outputdir=$4
bashpid=$$
cp ${templatefile} tmp.${bashpid}.py
sed -i "s/<nosplits>/$nosplits/g" tmp.${bashpid}.py
sed -i "s/<NCC_S>/$name/g" tmp.${bashpid}.py
mv tmp.${bashpid}.py ${outputdir}/${name}.py