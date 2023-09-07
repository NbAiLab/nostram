#!/usr/bin/env bash

NJ=2

ln -sf $(readlink -f kaldi_models) work/exp
cp utils/path.sh work
cd work

. ./path.sh

ln -sf ${KALDI_ROOT}/egs/wsj/s5/conf
ln -sf ${KALDI_ROOT}/egs/wsj/s5/steps
ln -sf ${KALDI_ROOT}/egs/wsj/s5/local
ln -sf ${KALDI_ROOT}/egs/wsj/s5/utils

./steps/make_mfcc.sh --nj $NJ --mfcc-config conf/mfcc_hires.conf data
./steps/compute_cmvn_stats.sh data
 ./steps/online/nnet2/extract_ivectors_online.sh --nj $NJ data exp/nnet3/extractor ivectors

cut -f2- -d' ' data/text | tr ' ' '\n' | sort -u > word.list

mkdir -p dict
phonetisaurus-apply --model exp/phonetisaurus-hr/model.fst --word_list word.list --nbest 10 --pmass 0.8 > dict/lexicon.txt
echo "<unk> spn" >> dict/lexicon.txt
sort -o dict/lexicon.txt dict/lexicon.txt
printf "L S Z a b d dZ dzp e f g i j k l m n nj o p r s t tS tcp ts u v x z\nsil sp spn\n" > dict/extra_question.txt
printf "sil\nsp\nspn\n" > dict/silence_phones.txt
printf "L\nS\nZ\na\nb\nd\ndZ\ndzp\ne\nf\ng\ni\nj\nk\nl\nm\nn\nnj\no\np\nr\ns\nt\ntS\ntcp\nts\nu\nv\nx\nz\n" > dict/nonsilence_phones.txt
printf "sp\n" > dict/optional_silence.txt

./utils/prepare_lang.sh dict "<unk>" tmp lang

./steps/nnet3/align.sh --nj 2 --use-gpu false --online-ivector-dir ivectors --beam 100 --retry-beam 400 data lang exp/nnet3/tdnn1a_sp ali

./steps/get_train_ctm.sh --use-segments true data lang ali
