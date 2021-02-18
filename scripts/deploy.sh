#!/usr/bin/env bash
if [[ "$#" == 0 ]]; then
	HOST="localhost"
elif [[ "$#" == 1 ]]; then
	HOST=$1
else
	echo "Wrong syntax! Usage: deploy.sh 'hostname'"
	exit 1
fi

SRC_FILE="vortector.py"

# get the scripts location and cd to it
SCRIPT_DIR=$(python3 -c "import os; print(os.path.dirname(os.path.realpath('$0')))")
cd $SCRIPT_DIR
cd ..

if [[ ! -e "$SRC_FILE" || ! -e "setup.py" ]]; then
	echo "Could not find $SRC_FILE or setup.py. Make sure they exist!"
	exit 1
fi
CODENAME=$(python3 -c "import os; print(os.path.basename(os.path.realpath('.')))")

UUID=$(python3 -c "import uuid; print(uuid.uuid4())")

# handle platforms without /tmp but TMPDIR set
if [[ "$HOST" != "localhost" ]]; then
	TMPDIR="$(ssh $HOST sh -c 'echo $TMPDIR')"
fi
REMOTE_TMP_DIR="$([[ -n $TMPDIR ]] && echo $TMPDIR || echo /tmp)"
REMOTE_TMP_DIR="$REMOTE_TMP_DIR/$UUID-install-$CODENAME"

echo "Deploying '$CODENAME' on '$HOST'"
if [[ "$HOST" == "localhost" ]]; then
	# copy files
	rsync setup.py $SRC_FILE $REMOTE_TMP_DIR
	# install and clean up
	sh -c "cd $REMOTE_TMP_DIR; python3 setup.py install --user; cd -; rm -rf $REMOTE_TMP_DIR" 1>/dev/null
else
	# copy files
	rsync $SRC_FILE setup.py $HOST:$REMOTE_TMP_DIR
	# install and clean up
	ssh $HOST "cd $REMOTE_TMP_DIR; python3 setup.py install --user; cd -; rm -rf $REMOTE_TMP_DIR" 1>/dev/null
fi
