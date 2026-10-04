#!/bin/bash
# Once arm ent0.01 is done, stop the be3dbe0 sweep (which will just have
# started ent0.1) and continue the remaining arms on 6989c7b, which is
# be3dbe0 plus the bit-identical rotor vectorisation only.
R=/home/hundo/Projects/Dytiscidae/dytiscidae/runs
until [ -f $R/r_ent0.01/done ]; do sleep 5; done
systemctl --user stop r-sweep-1003
sleep 5
# The just-started ent0.1 arm on the old tree: discard its seconds of output.
[ -f $R/r_ent0.1/done ] || rm -rf $R/r_ent0.1 $R/r_ent0.1.log
echo "$(date -Is) handover to 6989c7b" >> $R/_pass1003/r_sweep.status
systemd-run --user --unit r-sweep2-1003 -p MemoryMax=6500M /bin/bash $R/_pass1003/r_sweep2.sh
