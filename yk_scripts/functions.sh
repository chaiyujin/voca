#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

source yk_scripts/lib.sh

CWD=${PWD}
TALK_VIDEO_ROOT=$(realpath "${CWD}/../../stylized-sa/data/datasets/talk_video")

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                Collection of steps                                               * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function RUN_YK_EXP() {
  local DATA_SRC=
  local SPEAKER=
  local AVOFFSET_MS=
  local EPOCH=
  local MEDIA_LIST=
  local TEST=
  local LOAD_STEP=
  local DEBUG=""
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_src=*   ) DATA_SRC=${var#*=}    ;;
      --speaker=*    ) SPEAKER=${var#*=}     ;;
      --avoffset_ms=*) AVOFFSET_MS=${var#*=} ;;
      --epoch=*      ) EPOCH=${var#*=}       ;;
      --media_list=* ) MEDIA_LIST=${var#*=}  ;;
      --load_step=*  ) LOAD_STEP=${var#*=}   ;;
      --test         ) TEST="true"           ;;
      --debug        ) DEBUG="--debug"       ;;
    esac
  done
  # Check variables
  [ -n "$DATA_SRC"  ] || { echo "data_src is not set!";   exit 1; }
  [ -n "$SPEAKER"   ] || { echo "speaker is not set!";   exit 1; }
  # to lower case
  DATA_SRC="${DATA_SRC,,}"
  # to abspath
  if [ -n "$MEDIA_LIST" ] && [ "${MEDIA_LIST:0:1}" != "/" ]; then
    MEDIA_LIST=${CWD}/${MEDIA_LIST}
  fi

  # other variables
  local EXP_DIR="$CWD/yk_exp/$DATA_SRC/$SPEAKER"
  if [ -n "${AVOFFSET_MS}" ]; then
    EXP_DIR="${EXP_DIR}/avoffset_corrected"
  fi

  local NET_DIR="$EXP_DIR/training"
  local RES_DIR="$EXP_DIR/results"
  local DATA_DIR="$EXP_DIR/data"

  # Print arguments
  DRAW_DIVIDER;
  printf "Speaker   : $SPEAKER\n"
  printf "Data dir  : $DATA_DIR\n"
  printf "Ckpt dir  : $NET_DIR\n"
  printf "Results   : $RES_DIR\n"

  if [ -n "${AVOFFSET_MS}" ]; then
    # * Step 0: Correct vocaset avoffset
    DRAW_DIVIDER;
    python3 -m yk_scripts.correct_vocaset_avoffset;

    # * Step 1: Prepare data
    DRAW_DIVIDER;
    RUN_WITH_LOCK_GUARD --tag="Data" --lock_file=$EXP_DIR/done_data.lock -- \
      python3 -m yk_scripts.process_data --data_src=$DATA_SRC --speaker=$SPEAKER --avoffset_ms=$AVOFFSET_MS --data_dir=${DATA_DIR};
    
    # * Step 2: Train model
    if [ -n "$EPOCH" ]; then
      DRAW_DIVIDER;
      RUN_WITH_LOCK_GUARD --tag="Train" --lock_file=$EXP_DIR/done_train_${EPOCH}.lock -- \
        python3 -m yk_scripts.train \
          --exp_dir=${EXP_DIR} --net_dir=${NET_DIR} --epoch ${EPOCH} \
          --speaker=${SPEAKER} --correct_vocaset_avoffset;
    fi

  else
    # * Step 1: Prepare data
    DRAW_DIVIDER;
    RUN_WITH_LOCK_GUARD --tag="Data" --lock_file=$EXP_DIR/done_data.lock -- \
      python3 -m yk_scripts.process_data --speaker=$SPEAKER --data_src=$DATA_SRC;

    # * Step 2: Train model
    if [ -n "$EPOCH" ]; then
      DRAW_DIVIDER;
      RUN_WITH_LOCK_GUARD --tag="Train" --lock_file=$EXP_DIR/done_train_${EPOCH}.lock -- \
        python3 -m yk_scripts.train --exp_dir=${EXP_DIR} --net_dir=${NET_DIR} --epoch ${EPOCH} --speaker=${SPEAKER};
    fi
  fi

  # * Step 3: Test trained model
  if [ -n "$TEST" ]; then
    local CKPT="$NET_DIR/checkpoints/gstep_${LOAD_STEP}.model";
    local TMPL="$EXP_DIR/template.ply"
    local SHARED="--tf_model_fname=$CKPT --template_fname=$TMPL --condition_idx 1 --visualize=False"

    # dataset
    for d in "$TALK_VIDEO_ROOT/${DATA_SRC}/data/${SPEAKER}"/*; do
      if [ ! -f "$d/audio.wav" ]; then continue; fi
      local seq_id="$(basename $d)"
      python3 run_voca.py $SHARED \
        --audio_fname=${d}/audio.wav \
        --out_path="${RES_DIR}/clip-${seq_id}" \
      ;
    done

    # media_list if given
    if [ -n "${MEDIA_LIST}" ]; then
      local media_list=$(LoadMediaList ${MEDIA_LIST});
      for media_info in $media_list; do
        IFS='|' read -ra ADDR <<< "$media_info"
        local fpath="${ADDR[0]}"
        local seq_id="${ADDR[1]}"
        python3 run_voca.py $SHARED \
          --audio_fname="$fpath" \
          --out_path="${RES_DIR}/${seq_id}" \
        ;
      done
    fi
  fi
}
