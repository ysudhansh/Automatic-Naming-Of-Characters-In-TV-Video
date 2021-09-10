import pysrt 
import re

def getnames(srtfile,fps):
    subtitles = pysrt.open(srtfile)
    speakerlist = []
    for i in subtitles:
        starttime = float(i.start.minutes * 60.0 + i.start.seconds *1.0 + i.start.milliseconds*0.001)
        endtime = float(i.end.minutes * 60.0 + i.end.seconds *1.0 + i.end.milliseconds*0.001)
        #fps should be closest integer, int() will give floor. adding 0.49 so that anything greater than .5 will give ceil
        startframe = int(fps*starttime+0.49)
        endframe = int(fps*endtime+0.49)
        speaker = re.search(r"\[([A-Za-z0-9_]+)\]", i.text)
        if speaker != None:
            speaker = speaker.group(1)
        for j in range(len(speakerlist),startframe):
            speakerlist.append(None)
        for j in range(startframe,endframe):
            speakerlist.append(speaker)
    return speakerlist


if __name__ =='__main__':
    temp = getnames('subtitles.srt',25)
    print(temp)
