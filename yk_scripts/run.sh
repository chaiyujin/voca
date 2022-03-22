LIB=yk_scripts/functions.sh
source ${LIB}
source yk_scripts/nohup_run.sh

# > Keep avoffset unchanged
# RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump   --epoch=50 --load_step=100400 "$@";

# > Manually correct avoffset
# RUN_YK_EXP --data_src=celebtalk --speaker=m001_trump   --avoffset_ms=33.333333333333336 --epoch=50 --load_step=98200 "$@";
# RUN_YK_EXP --data_src=celebtalk --speaker=m000_obama   --avoffset_ms=100.0              --epoch=50 --load_step=97250 "$@";
# RUN_YK_EXP --data_src=celebtalk --speaker=f000_watson  --avoffset_ms=133.33333333333334 --epoch=50 --load_step=95900 "$@";
# RUN_YK_EXP --data_src=celebtalk --speaker=f001_clinton --avoffset_ms=100.0              --epoch=50 --load_step=98500 "$@";
