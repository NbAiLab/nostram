#!/bin/bash
if test $# -ne 10 
then
echo "Usage:$0 <dataset> <outputdir> "
#echo "${10}" > aa.txt
exit
fi
#"echo "${10}" > aa.txt
templatefile=$1
datasetname=$2
versionnumber=$3
nrkduration=$5
nrkclips=$4
nstduration=$7
nstclips=$6
npscduration=$9
npscclips=$8
outputfile=${10}




bashpid=$$
#echo ${templatefile}
#echo ${outputdir}


cp ${templatefile} tmp.${bashpid}.txt
sed -i "s/<dataset_name>/${datasetname}/g" tmp.${bashpid}.txt
sed -i "s/<versionnumber>/${versionnumber}/g" tmp.${bashpid}.txt
if test ${nrkduration} -eq 0
then
sed -i '/<nrkduration>/d' tmp.${bashpid}.txt
else
nrkminutes=`echo "scale=2; ${nrkduration} / 3600000" | bc `
sed -i "s/<nrkduration>/${nrkminutes}/g" tmp.${bashpid}.txt
sed -i "s/<nrkclips>/${nrkclips}/g" tmp.${bashpid}.txt
fi
if test ${nstduration} -eq 0
then
sed -i '/<nstduration>/d' tmp.${bashpid}.txt
else
nstminutes=`echo "scale=2; ${nstduration} / 3600000" | bc `
sed -i "s/<nstduration>/${nstminutes}/g" tmp.${bashpid}.txt
sed -i "s/<nstclips>/${nstclips}/g" tmp.${bashpid}.txt
fi
if test ${npscduration} -eq 0
then
sed -i '/<npscduration>/d' tmp.${bashpid}.txt
else
npscminutes=`echo "scale=2; ${npscduration} / 3600000" | bc `
sed -i "s/<npscduration>/${npscminutes}/g" tmp.${bashpid}.txt
sed -i "s/<npscclips>/${npscclips}/g" tmp.${bashpid}.txt
fi

mv tmp.${bashpid}.txt ${outputfile}
