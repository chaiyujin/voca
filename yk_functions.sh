#!/bin/bash
set -o errexit
set -o pipefail
set -o nounset

CWD=${PWD}
TALK_VIDEO_ROOT=${CWD}/../../stylized-sa/data/datasets/talk_video

function LoadMediaList() {
  local filepath=$1;
  shift 1;

  input=$filepath
  while IFS= read -r line
  do
    # trim
    line=$(echo $line | xargs)
    # valid
    if [ -n "$line" ] && [ "${line:0:1}" != "#" ]; then
      if [ "${line:0:1}" != "/" ]; then
        line=$(dirname $filepath)/${line}
      fi
      echo "$line"
    fi
  done < "$input"
}

function DRAW_DIVIDER() {
  printf '%*s\n' "${COLUMNS:-$(tput cols)}" ' ' | tr ' ' '-'
}

function RUN_WITH_LOCK_GUARD() {
  local _end=
  local lock_file=
  local tag=
  local debug=
  local cmd=
  for i in "$@"; do
    if [ -z "${_end}" ]; then
      if [ "$i" = "--" ]; then
        _end=true
        continue
      fi
      case $i in
        -l=* | --lock_file=* ) lock_file=${i#*=}  ;;
        -t=* | --tag=*       ) tag=${i#*=}        ;;
        -d   | --debug       ) debug=true         ;;
        *) echo "[LockFileGuard]: Wrong argument ${i}"; exit 1;;
      esac
    else
      cmd="$cmd '$i'"
    fi
  done

  # lock_file must be given
  if [ -z "$lock_file" ]; then
    printf "lock_file is not set!\n"
    exit 1
  fi

  if [ -f "$lock_file" ]; then
    printf "Command '${tag}' is skipped due to existance of lock_file ${lock_file}\n"
    return
  fi

  # to abspath
  mkdir -p "$(dirname $lock_file)"
  lock_file="$(cd $(dirname $lock_file) && pwd)/$(basename $lock_file)"

  # debug message
  if [ -n "$debug" ]; then
    echo "--------------------------------------------------------------------------------"
    echo "LOCK_FILE: $lock_file"
    echo "COMMAND:   $cmd"
    echo "--------------------------------------------------------------------------------"
  fi

  # Run command in subshell and creat lock file if success
  if (eval "$cmd") ; then
    [ -f "$lock_file" ] || touch "$lock_file"
  else
    echo "Failed to run command '${tag}'"
    exit 1
  fi
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                       Data                                                       * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function PrepareData() {
  local DATA_DIR=
  local DATA_SRC=
  local SPEAKER=
  local USE_SEQS=
  local DEBUG=
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_dir=* ) DATA_DIR=${var#*=}  ;;
      --data_src=* ) DATA_SRC=${var#*=}  ;;
      --speaker=*  ) SPEAKER=${var#*=}   ;;
      --use_seqs=* ) USE_SEQS=${var#*=}  ;;
      --debug      ) DEBUG="--debug"     ;;
    esac
  done

  [ -n "$DATA_DIR" ] || { echo "data_dir is not set!"; exit 1; }
  [ -n "$DATA_SRC" ] || { echo "data_src is not set!"; exit 1; }
  [ -n "$SPEAKER"  ] || { echo "speaker is not set!";  exit 1; }

  # prepare data
  RUN_WITH_LOCK_GUARD --tag="Data" --lock_file=$DATA_DIR/../done_data.lock -- \
    python3 yk_process_data.py --speaker=$SPEAKER --data_src=$DATA_SRC --use_seqs=${USE_SEQS};
}

# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                Collection of steps                                               * #
# * ---------------------------------------------------------------------------------------------------------------- * #

function RUN_YK_EXP() {
  local DATA_SRC=
  local SPEAKER=
  local USE_SEQS=
  local EPOCH=
  local MEDIA_LIST=
  local TEST=
  local LOAD_STEP=
  local DEBUG=""
  # Override from arguments
  for var in "$@"; do
    case $var in
      --data_src=*  ) DATA_SRC=${var#*=}   ;;
      --speaker=*   ) SPEAKER=${var#*=}    ;;
      --use_seqs=*  ) USE_SEQS=${var#*=}   ;;
      --epoch=*     ) EPOCH=${var#*=}      ;;
      --media_list=*) MEDIA_LIST=${var#*=} ;;
      --load_step=* ) LOAD_STEP=${var#*=}  ;;
      --test        ) TEST="true"          ;;
      --debug       ) DEBUG="--debug"      ;;
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
  local NET_DIR="$EXP_DIR/checkpoints"
  local RES_DIR="$EXP_DIR/results"
  local DATA_DIR="$EXP_DIR/data"

  # Print arguments
  DRAW_DIVIDER;
  printf "Speaker   : $SPEAKER\n"
  printf "Data dir  : $DATA_DIR\n"
  printf "Ckpt dir  : $NET_DIR\n"
  printf "Results   : $RES_DIR\n"

  DRAW_DIVIDER;
  PrepareData \
    --data_dir=${DATA_DIR} \
    --data_src=$DATA_SRC \
    --speaker=$SPEAKER \
    --use_seqs=$USE_SEQS \
    ${DEBUG} \
  ;

  if [ -n "$EPOCH" ]; then
    DRAW_DIVIDER;
    RUN_WITH_LOCK_GUARD --tag="Train" --lock_file=$EXP_DIR/../done_train_${EPOCH}.lock -- \
      python3 yk_train.py --exp_dir=${EXP_DIR} --epoch ${EPOCH};
  fi

  if [ -n "$TEST" ]; then
    local CKPT="$EXP_DIR/training/checkpoints/gstep_${LOAD_STEP}.model";
    local TMPL="$EXP_DIR/template.ply"
    local SHARED="--tf_model_fname=$CKPT --template_fname=$TMPL --condition_idx 1 --visualize=False"

    for d in "$TALK_VIDEO_ROOT/${DATA_SRC}/data/${SPEAKER}"/*; do
      if [ ! -f "$d/audio.wav" ]; then continue; fi
      local seq_id="$(basename $d)"
      python3 run_voca.py $SHARED \
        --audio_fname=${d}/audio.wav \
        --out_path="${RES_DIR}/clip-${seq_id}" \
      ;
    done

    if [ -n "${MEDIA_LIST}" ]; then
      local media_list=$(LoadMediaList ${MEDIA_LIST});
      for media_path in $media_list; do
        local seq_id="$(basename $media_path)"
        seq_id="${seq_id%.*}"
        python3 run_voca.py $SHARED \
          --audio_fname="$media_path" \
          --out_path="${RES_DIR}/${seq_id}" \
        ;
      done
    fi
  fi
}
