#!/usr/bin/env bash

SIMID="$1"

if [[ -e "videos/$SIMID" ]];
then
	echo "Video exists. Delete folder to create again."
	exit 0
fi

function get_output_number() {
	SIMID="$1"
	SIMDIR=$(smurf search $SIMID -p path)
	SIMHOST=$(smurf search $SIMID -p host)

	if [[ "$SIMHOST" == "localhost" ]];
	then
		N=$(ls $SIMDIR/output 2>/dev/null | grep gasdens | sort -V | tail -n 1 | grep -oP "[0-9]+")
	else
		N=$(ssh $SIMHOST ls $SIMDIR/output 2>/dev/null | grep gasdens | sort -V | tail -n 1 | grep -oP "[0-9]+")
	fi


	if [[ -z "$N" ]];
	then
		if [[ "$SIMHOST" == "localhost" ]];
		then
			N=$(ls $SIMDIR/data/out 2>/dev/null | grep data | sort -V | tail -n 1 | grep -oP "[0-9]+")
		else
			N=$(ssh $SIMHOST ls $SIMDIR/data/out 2>/dev/null | grep data | sort -V | tail -n 1 | grep -oP "[0-9]+")
		fi
	fi
	if [[ -z "$N" ]];
	then
		if [[ "$SIMHOST" == "localhost" ]];
		then
			N=$(ls $SIMDIR/data 2>/dev/null | grep data | sort -V | tail -n 1 | grep -oP "[0-9]+")
		else
			N=$(ssh $SIMHOST ls $SIMDIR/data 2>/dev/null | grep data | sort -V | tail -n 1 | grep -oP "[0-9]+")
		fi
	fi
	if [[ -z "$N" ]];
	then
		echo "Could not find output number"
		exit 1
	fi
	echo $N
}
N=$(get_output_number $SIMID)

if [[ $? -eq 1 ]];
then
        echo "$N"
        exit 1
fi

echo "Animating $N frames"

mkdir -p "videos/$SIMID"

seq 0 $N | xargs -n 1 -P 6 -I {} ./simdata_vortector.py -v $SIMID {} --outfile videos/$SIMID/{}.jpg --nocache

echo "Making video for $SIMID"
dipper make_video "videos/$SIMID"
