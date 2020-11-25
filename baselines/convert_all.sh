#!/bin/bash


XQUAD_LANG="ar de el en es hi ru th tr vi zh"
TYDI_LANG="th sw te fi be ru ja ar in ko en"
MLQA_LANG="ar de en es hi vi zh"


for LANG in $MLQA_LANG
do
    python convert_corpora.py ../../MLQA_V1 ../../data -f mlqa -p dev -l $LANG
    python convert_corpora.py ../../MLQA_V1 ../../data -f mlqa -p test -l $LANG
done

for LANG in $TYDI_LANG
do
    python convert_corpora.py ../../ ../../data -f tydi -p dev -l $LANG
    python convert_corpora.py ../../ ../../data -f tydi -p train -l $LANG
done

for LANG in $XQUAD_LANG
do
    python convert_corpora.py ../../xquad-master ../../data -f xquad -p dev -l $LANG
    python convert_corpora.py ../../xquad-master ../../data -f xquad-context -p dev -l $LANG
done

echo "finished"

