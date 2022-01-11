# PATH=/bin:/sbin:/usr/bin:/usr/sbin:/usr/local/bin:/usr/local/sbin:~/bin:/usr/local/ffmpeg/bin
# export PATH

filelist=$(find . -name '*.mp3' -o -name '*.wav')
OLDIFS="$IFS"
IFS=$"\n"
for filename in *.mp3 *.wav
	do
		if [$filename == '.DS_Store' || $filename == ffmpeg_foo_to_wav.sh]
		then
			continue
		fi
		ffmpeg -i $filename -acodec pcm_s16le -ar 16000 -ac 1 wav/"${filename%.*}"-lpb.wav
	done
IFS=$OLDIFS
