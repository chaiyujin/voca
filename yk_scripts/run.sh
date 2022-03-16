source yk_scripts/nohup_run.sh
source yk_scripts/functions.sh

RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump --epoch=50 --load_step=100400 "$@";
