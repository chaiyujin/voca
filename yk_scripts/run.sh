LIB=yk_scripts/functions.sh
source ${LIB}
source yk_scripts/nohup_run.sh

# * Keep avoffset unchanged
# RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump --epoch=50 --load_step=100400 "$@";

# * Manually correct avoffset
# RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump --epoch=50 --avoffset_ms=33.333333333333336 --load_step=98200 "$@";
