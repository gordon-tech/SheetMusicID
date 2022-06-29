
class Config:
    # Pre-processing
    thumbnailW = 100  # bkgd lighting
    thumbnailH = 100
    thumbnailFilterSize = 5
    estLineSep_NumCols = 3
    # 五线谱的线间距必须在[estLineSep_LowerRange, estLineSep_UpperRange)之间
    estLineSep_LowerRange = 8
    estLineSep_UpperRange = 50
    estLineSep_Delta = 1
    targetLineSep = 10.0

    # Staff Line Features
    morphFilterHorizLineSize = 15
    notebarFiltLen = 3
    notebarRemoval = 0.9
    calcStaveFeatureMap_NumCols = 10
    calcStaveFeatureMap_LowerRange = 8.5
    calcStaveFeatureMap_UpperRange = 11.75
    calcStaveFeatureMap_Delta = 0.125

    # Staffline Detection
    maxDeltaRowInitial = 50
    minNumStaves = 2
    maxNumStaves = 20
    minStaveSeparation = 6 * targetLineSep
    maxDeltaRowRefined = 15

    # Group Staves
    morphFilterVertLineLength = 35
    morphFilterVertLineWidth = 10
    maxBarlineWidth = 40
    ##########################

opt = Config()