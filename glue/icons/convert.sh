#!/bin/bash

for input in *.svg
do
    output=${input/svg/png}
    inkscape --without-gui --export-png=$output --export-width=125 --export-height=125 $input
done

convert -flop playback_forw.png playback_back.png
convert -flop playback_next.png playback_prev.png
convert -flop playback_last.png playback_first.png