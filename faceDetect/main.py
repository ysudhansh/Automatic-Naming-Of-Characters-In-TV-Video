from face_detect import face_detect
from video_detect import video_detect, video_cap
from ext_ele import ext_ele
from process_subtitles import getnames
import time
from imutils.video import FileVideoStream

def main ():
    # fvs = FileVideoStream("office.mp4").start()
    # time.sleep(1.0)
    # # frame_count = 0
    # # df_pos = 0
    # while fvs.more():
    #     frm = fvs.read()
    #     face_detect(frm)
    #face_detect("office.mp4")
    # characters = individual_faces("./chars")
    subs = getnames("subtitles.srt", 25)
    face_detect("standoff.mp4", subs)
    # video_detect("./office.mp4")
    # video_cap(1)
    # ext_ele()


if __name__ == '__main__':
    main()

