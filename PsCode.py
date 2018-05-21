def get_ps_code(padding_to_apply):

    string_lasso_begin = "var idsetd = charIDToTypeID( \"setd\" );\n\
        var desc1 = new ActionDescriptor();\n\
        var idnull = charIDToTypeID( \"null\" );\n\
            var ref28 = new ActionReference();\n\
            var idChnl = charIDToTypeID( \"Chnl\" );\n\
            var idfsel = charIDToTypeID( \"fsel\" );\n\
            ref28.putProperty( idChnl, idfsel );\n\
        desc1.putReference( idnull, ref28 );\n\
        var idT = charIDToTypeID( \"T   \" );\n\
            var desc2 = new ActionDescriptor();\n\
            var idPts = charIDToTypeID( \"Pts \" );\n\
                var list6 = new ActionList();\n"

    string_lasso_end = "        desc2.putList( idPts, list6 );\n\
        var idPlgn = charIDToTypeID( \"Plgn\" );\n\
        desc1.putObject( idT, idPlgn, desc2 );\n\
        var idAntA = charIDToTypeID( \"AntA\" );\n\
        desc1.putBoolean( idAntA, true );\n\
    executeAction( idsetd, desc1, DialogModes.NO );\n"
    # Need to get size of image and add pixels based on ation, maybe 1% of pixel width/height


    # Executes contentaware
    content_aware_fill = "var idFl = charIDToTypeID( \"Fl  \" );\n\
        var desc30 = new ActionDescriptor();\n\
        var idUsng = charIDToTypeID( \"Usng\" );\n\
        var idFlCn = charIDToTypeID( \"FlCn\" );\n\
        var idcontentAware = stringIDToTypeID( \"contentAware\" );\n\
        desc30.putEnumerated( idUsng, idFlCn, idcontentAware );\n\
        var idOpct = charIDToTypeID( \"Opct\" ); \n\
        var idPrc = charIDToTypeID( \"#Prc\" ); \n\
        desc30.putUnitDouble( idOpct, idPrc, 100.000000 );\n\
        var idMd = charIDToTypeID( \"Md  \" );\n\
        var idBlnM = charIDToTypeID( \"BlnM\" );\n\
        var idNrml = charIDToTypeID( \"Nrml\" );\n\
        desc30.putEnumerated( idMd, idBlnM, idNrml );\n\
    executeAction( idFl, desc30, DialogModes.NO );\n "

    # Expand padding by pixel size here = 10
    add_padding = "var idExpn = charIDToTypeID( \"Expn\" );\n\
        var desc54 = new ActionDescriptor();\n\
        var idBy = charIDToTypeID( \"By  \" );\n\
        var idPxl = charIDToTypeID( \"#Pxl\" );\n\
        desc54.putUnitDouble( idBy, idPxl, " + str(padding_to_apply) + " );\n\
    executeAction( idExpn, desc54, DialogModes.NO );\n"

    create_mask = "var idMk = charIDToTypeID( \"Mk  \" );\n\
        var desc29 = new ActionDescriptor();\n\
        var idNw = charIDToTypeID( \"Nw  \" );\n\
        var idChnl = charIDToTypeID( \"Chnl\" );\n\
        desc29.putClass( idNw, idChnl );\n\
        var idAt = charIDToTypeID( \"At  \" );\n\
            var ref20 = new ActionReference();\n\
            var idChnl = charIDToTypeID( \"Chnl\" );\n\
            var idChnl = charIDToTypeID( \"Chnl\" );\n\
            var idMsk = charIDToTypeID( \"Msk \" );\n\
            ref20.putEnumerated( idChnl, idChnl, idMsk );\n\
        desc29.putReference( idAt, ref20 );\n\
        var idUsng = charIDToTypeID( \"Usng\" );\n\
        var idUsrM = charIDToTypeID( \"UsrM\" );\n\
        var idRvlA = charIDToTypeID( \"RvlA\" );\n\
        desc29.putEnumerated( idUsng, idUsrM, idRvlA );\n\
    executeAction( idMk, desc29, DialogModes.NO );\n"

    deselect_mask = "var idsetd = charIDToTypeID( \"setd\" );\n\
        var desc99 = new ActionDescriptor();\n\
        var idnull = charIDToTypeID( \"null\" );\n\
            var ref72 = new ActionReference();\n\
            var idChnl = charIDToTypeID( \"Chnl\" );\n\
            var idfsel = charIDToTypeID( \"fsel\" );\n\
            ref72.putProperty( idChnl, idfsel );\n\
        desc99.putReference( idnull, ref72 );\n\
        var idT = charIDToTypeID( \"T   \" );\n\
        var idOrdn = charIDToTypeID( \"Ordn\" );\n\
        var idNone = charIDToTypeID( \"None\" );\n\
        desc99.putEnumerated( idT, idOrdn, idNone );\n\
    executeAction( idsetd, desc99, DialogModes.NO );\n"

    select_image_not_mask = "var idslct = charIDToTypeID( \"slct\" );\n\
        var desc23 = new ActionDescriptor();\n\
        var idnull = charIDToTypeID( \"null\" );\n\
            var ref14 = new ActionReference();\n\
            var idChnl = charIDToTypeID( \"Chnl\" );\n\
            var idChnl = charIDToTypeID( \"Chnl\" );\n\
            var idRGB = charIDToTypeID( \"RGB \" );\n\
            ref14.putEnumerated( idChnl, idChnl, idRGB );\n\
        desc23.putReference( idnull, ref14 );\n\
        var idMkVs = charIDToTypeID( \"MkVs\" );\n\
        desc23.putBoolean( idMkVs, false );\n\
    executeAction( idslct, desc23, DialogModes.NO )\n"

    mask_fill_black = "var idFl = charIDToTypeID( \"Fl  \" );\n\
        var desc18 = new ActionDescriptor();\n\
        var idUsng = charIDToTypeID( \"Usng\" );\n\
        var idFlCn = charIDToTypeID( \"FlCn\" );\n\
        var idClr = charIDToTypeID( \"Clr \" );\n\
        desc18.putEnumerated( idUsng, idFlCn, idClr );\n\
        var idClr = charIDToTypeID( \"Clr \" );\n\
            var desc19 = new ActionDescriptor();\n\
            var idH = charIDToTypeID( \"H   \" );\n\
            var idAng = charIDToTypeID( \"#Ang\" );\n\
            desc19.putUnitDouble( idH, idAng, 0.000000 );\n\
            var idStrt = charIDToTypeID( \"Strt\" );\n\
            desc19.putDouble( idStrt, 0.000000 );\n\
            var idBrgh = charIDToTypeID( \"Brgh\" );\n\
            desc19.putDouble( idBrgh, 0.000000 );\n\
        var idHSBC = charIDToTypeID( \"HSBC\" );\n\
        desc18.putObject( idClr, idHSBC, desc19 );\n\
        var idOpct = charIDToTypeID( \"Opct\" );\n\
        var idPrc = charIDToTypeID( \"#Prc\" );\n\
        desc18.putUnitDouble( idOpct, idPrc, 100.000000 );\n\
        var idMd = charIDToTypeID( \"Md  \" );\n\
        var idBlnM = charIDToTypeID( \"BlnM\" );\n\
        var idNrml = charIDToTypeID( \"Nrml\" );\n\
        desc18.putEnumerated( idMd, idBlnM, idNrml );\n\
    executeAction( idFl, desc18, DialogModes.NO );\n"

    disable_layer_mask = "var idsetd = charIDToTypeID( \"setd\" );\n\
        var desc185 = new ActionDescriptor();\n\
        var idnull = charIDToTypeID( \"null\" );\n\
            var ref139 = new ActionReference();\n\
            var idLyr = charIDToTypeID( \"Lyr \" );\n\
            var idOrdn = charIDToTypeID( \"Ordn\" );\n\
            var idTrgt = charIDToTypeID( \"Trgt\" );\n\
            ref139.putEnumerated( idLyr, idOrdn, idTrgt );\n\
        desc185.putReference( idnull, ref139 );\n\
        var idT = charIDToTypeID( \"T   \" );\n\
            var desc186 = new ActionDescriptor();\n\
            var idUsrM = charIDToTypeID( \"UsrM\" );\n\
            desc186.putBoolean( idUsrM, false );\n\
        var idLyr = charIDToTypeID( \"Lyr \" );\n\
        desc185.putObject( idT, idLyr, desc186 );\n\
    executeAction( idsetd, desc185, DialogModes.NO );\n"

    Try_layer_from_background = "try{\n\
    var idsetd = charIDToTypeID( \"setd\" );\n\
        var desc30 = new ActionDescriptor();\n\
        var idnull = charIDToTypeID( \"null\" );\n\
            var ref22 = new ActionReference();\n\
            var idLyr = charIDToTypeID( \"Lyr \" );\n\
            var idBckg = charIDToTypeID( \"Bckg\" );\n\
            ref22.putProperty( idLyr, idBckg );\n\
        desc30.putReference( idnull, ref22 );\n\
        var idT = charIDToTypeID( \"T   \" );\n\
            var desc31 = new ActionDescriptor();\n\
            var idOpct = charIDToTypeID( \"Opct\" );\n\
            var idPrc = charIDToTypeID( \"#Prc\" );\n\
            desc31.putUnitDouble( idOpct, idPrc, 100.000000 );\n\
            var idMd = charIDToTypeID( \"Md  \" );\n\
            var idBlnM = charIDToTypeID( \"BlnM\" );\n\
            var idNrml = charIDToTypeID( \"Nrml\" );\n\
            desc31.putEnumerated( idMd, idBlnM, idNrml );\n\
        var idLyr = charIDToTypeID( \"Lyr \" );\n\
        desc30.putObject( idT, idLyr, desc31 );\n\
    executeAction( idsetd, desc30, DialogModes.NO );\n\
    }\n\
    catch(e){}\n"

    save_photo = "var idsave = charIDToTypeID( \"save\" );\n\
        var desc2 = new ActionDescriptor();\n\
        var idAs = charIDToTypeID( \"As  \" );\n\
            var desc3 = new ActionDescriptor();\n\
            var idEQlt = charIDToTypeID( \"EQlt\" );\n\
            desc3.putInteger( idEQlt, 10 );\n\
            var idMttC = charIDToTypeID( \"MttC\" );\n\
            var idMttC = charIDToTypeID( \"MttC\" );\n\
            var idNone = charIDToTypeID( \"None\" );\n\
            desc3.putEnumerated( idMttC, idMttC, idNone );\n\
        var idJPEG = charIDToTypeID( \"JPEG\" );\n\
        desc2.putObject( idAs, idJPEG, desc3 );\n\
        var idIn = charIDToTypeID( \"In  \" );\n\
        desc2.putPath( idIn, new File( \"C:\\Users\\Paddy\\Documents\\FYP\\Mask_RCNN-master\\image8testscript.jpg\" ) );\n\
        var idDocI = charIDToTypeID( \"DocI\" );\n\
        desc2.putInteger( idDocI, 35 );\n\
        var idsaveStage = stringIDToTypeID( \"saveStage\" );\n\
        var idsaveStageType = stringIDToTypeID( \"saveStageType\" );\n\
        var idsaveBegin = stringIDToTypeID( \"saveBegin\" );\n\
        desc2.putEnumerated( idsaveStage, idsaveStageType, idsaveBegin );\n\
    executeAction( idsave, desc2, DialogModes.NO ); \n\
    var idsave = charIDToTypeID( \"save\" );\n\
        var desc4 = new ActionDescriptor();\n\
        var idAs = charIDToTypeID( \"As  \" );\n\
            var desc5 = new ActionDescriptor();\n\
            var idEQlt = charIDToTypeID( \"EQlt\" );\n\
            desc5.putInteger( idEQlt, 10 );\n\
            var idMttC = charIDToTypeID( \"MttC\" );\n\
            var idMttC = charIDToTypeID( \"MttC\" );\n\
            var idNone = charIDToTypeID( \"None\" );\n\
            desc5.putEnumerated( idMttC, idMttC, idNone );\n\
        var idJPEG = charIDToTypeID( \"JPEG\" );\n\
        desc4.putObject( idAs, idJPEG, desc5 );\n\
        var idIn = charIDToTypeID( \"In  \" );\n\
        desc4.putPath( idIn, new File( \"C:\\Users\\Paddy\\Documents\\FYP\\Mask_RCNN-master\\image8testscript.jpg\" ) );\n\
        var idDocI = charIDToTypeID( \"DocI\" );\n\
        desc4.putInteger( idDocI, 35 );\n\
        var idsaveStage = stringIDToTypeID( \"saveStage\" );\n\
        var idsaveStageType = stringIDToTypeID( \"saveStageType\" );\n\
        var idsaveSucceeded = stringIDToTypeID( \"saveSucceeded\" );\n\
        desc4.putEnumerated( idsaveStage, idsaveStageType, idsaveSucceeded );\n\
    executeAction( idsave, desc4, DialogModes.NO );\n"

    close_photoshop = "var idCls = charIDToTypeID( \"Cls \" );\n\
        executeAction( idCls, undefined, DialogModes.NO );\n"

    copy_to_newlayer = "var idCpTL = charIDToTypeID( \"CpTL\" );\n\
    executeAction( idCpTL, undefined, DialogModes.NO );\n"

    make_new_layer = "var idMk = charIDToTypeID( \"Mk  \" );\n\
    var desc1702 = new ActionDescriptor();\n\
    var idnull = charIDToTypeID( \"null\" );\n\
        var ref4 = new ActionReference();\n\
        var idLyr = charIDToTypeID( \"Lyr \" );\n\
        ref4.putClass( idLyr );\n\
    desc1702.putReference( idnull, ref4 );\n\
    executeAction( idMk, desc1702, DialogModes.NO );\n"

    return(string_lasso_begin,string_lasso_end,content_aware_fill,add_padding,create_mask,deselect_mask,select_image_not_mask,mask_fill_black,disable_layer_mask,Try_layer_from_background)

def pixel_selection(index,entry):
    string_to_print = "var desc" + str(index) + " = new ActionDescriptor();\n\
                                  var idHrzn = charIDToTypeID( \"Hrzn\" );\n\
                                  var idPxl = charIDToTypeID( \"#Pxl\" ); \n\
                                  desc" + str(index) + ".putUnitDouble( idHrzn, idPxl," + str(entry[1]) + ");\n\
                                  var idVrtc = charIDToTypeID( \"Vrtc\" );\n\
                                  var idPxl = charIDToTypeID( \"#Pxl\" );\n\
                                  desc" + str(index) + ".putUnitDouble( idVrtc, idPxl, " + str(entry[0]) + " );\n\
                              var idPnt = charIDToTypeID( \"Pnt \" );\n\
                                list6.putObject( idPnt, desc" + str(index) + " );\n"
    return(string_to_print)