#!/usr/bin/env python

import os
import sys
from python_get_resolve import GetResolve

def Export(timeline, filePath, exportType, exportSubType=None):
    result = None
    if exportSubType is None:
        result = timeline.Export(filePath, exportType)
    else:
        result = timeline.Export(filePath, exportType, exportSubType)

    if result:
        print("Timeline exported to {0} successfully.".format(filePath))
    else:
        print("Timeline export failed.")

def getFolder(parentFolder, childFolder, mp):
    for folder in parentFolder.GetSubFolderList():
        if folder.GetName() == childFolder:
            return folder
    else:
        return mp.AddSubFolder(parentFolder , childFolder)
    
if __name__ == "__main__":
    resolve = GetResolve()
    projectManager = resolve.GetProjectManager()
    project = projectManager.GetCurrentProject()
    mediaPool = project.GetMediaPool()
    rootFolder = mediaPool.GetRootFolder()
    editFolder = getFolder(rootFolder, "1990s", mediaPool)
    clips = editFolder.GetClipList()

    for clip in clips:
        if clip.GetClipProperty()["Video Codec"] != "":
            clipName = clip.GetName().removesuffix('.m4v')
            print(clipName)
            
            tl = mediaPool.CreateEmptyTimeline(clipName)
            mediaPool.AppendToTimeline(clip)
            tl.DetectSceneCuts()
            
            timeline = project.GetCurrentTimeline()
            xmlFilePath = os.path.join(os.path.expanduser("/Volumes/soul/Projects/detectedclips"), clipName + ".fcpxmld")
            Export(timeline, xmlFilePath, resolve.EXPORT_FCPXML_1_10)


