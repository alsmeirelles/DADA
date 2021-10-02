#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import pickle

if __name__ == "__main__":
    from TCGAParser import TCGABarcode
    from BCRParser import BCRParser
    from GenericData import ImageSource
    #from ..Utils import CacheManager
else:
    from .TCGAParser import TCGABarcode
    from .BCRParser import BCRParser
    from .GenericData import ImageSource
    from Utils import CacheManager

class Merger(ImageSource):
  """
  Builds a relation amongst cases and files
  """  
  def __init__(self,dataset=".",verbose=0):
    """
    Receives the path to a dataset as a parameter.
    The path should point to a directory containing file types compiled in dedicated directories:

    WSI: slide directory
    BCR: clinical information directory

    Main data is a dictionary (self._data), containing:
    key: case id
    value: dictionary with:
    ------ subkey: "wsi" -> list of TCGABarcode objects
    ------ subkey: "bcr" -> patient BCR object
    ------ subkey: "cdata" -> important case data (UNDER CONSIDERATION)
    """

    super().__init__(dataset,['svs'],verbose)

    self._data = None
    self._wsi = None
    self._gbm = None
    self._lgg = None
    self.run()

  def run(self):
    """
    Starts processing the dataset.
    """
    #Searchs for pickled data
    ls = os.listdir(self.path)
    
    try:
      ls.index("BCR")
      ls.index("WSI")
    except ValueError:
      if self._verbose > 0:
          print("[TCGAMerger - WSI and/or BCR directories not found in path: {0}".format(self.path))
          return None
      
    self._data = dict()
    self.generateBCR(ls)
        
    self._wsi = dict()
    self.generateWSI(ls)


  def getImgList(self):
    """
    Returns a list of SVSImage objects.
    """
    #TODO: change implementation to meet above definition
    return [SVSImage(p.getPath()) for p in self._wsi.values()]

  def produceImage(self):
    pass
  
  def generateBCR(self,filelist):
    """
    Start processing BCR files.
    """

    bcrlist = os.listdir(os.path.join(self.path,"BCR"))

    for f in bcrlist:
      case_files = os.listdir(os.path.join(self.path,"BCR",f))
      for item in case_files:
        if BCRParser.checkFileName(item):
          bcr_obj = BCRParser(os.path.join(self.path,"BCR",f,item))
          key = bcr_obj.returnPatientIDFromFile()
          if key in self._data:
            self._data[key]['bcr'].append(bcr_obj)
          else:
            self._data[key] = dict()
            self._data[key]['wsi'] = []
            self._data[key]['bcr'] = [bcr_obj]
            self._data[key]['cdata'] = dict()
    
  def generateWSI(self,filelist=[]):
    """
    Start processing WSI files.
    """
      
    wsilist = os.listdir(os.path.join(self.path,"WSI"))

    for f in wsilist:
      case_files = os.listdir(os.path.join(self.path,"WSI",f))
      has_annotation = False
      for item in case_files:
        if item == "annotations.txt":
          has_annotation = True
        if TCGABarcode.checkFileName(item):
          tcga = TCGABarcode(item,f)
          self._wsi[f] = tcga
          self._data[tcga.returnCaseID()]['wsi'].append(tcga)

      if has_annotation:
        self._wsi[f].setAnnotation()
        
  def addImageFile(self,uuid):
    """
    Add an image file to records.
    uuid corresponds to the image directory's uuid.
    """
    if uuid in self._wsi:
      print("Image already on record.")
      return

    if isinstance(uuid,str) and os.path.isdir(os.path.join(self.path,"WSI",uuid)):
      file_list = os.listdir(os.path.join(self.path,"WSI",uuid))

      has_annotation = False
      for f in file_list:
        if TCGABarcode.checkFileName(f):
          tcga = TCGABarcode(f,uuid)
          key = tcga.returnCaseID()
          if key in self._data:
            self._data[key]['wsi'].append(tcga)
            self._wsi[uuid] = tcga
          else:
            self._data[key] = dict()
            self._data[key]['bcr'] = self.searchBCR(tcga)
            self._data[key]['wsi'] = [tcga]
            self._wsi[uuid] = tcga
        if f == "annotations.txt":
          has_annotation = True

      if has_annotation:
        self._wsi[uuid].setAnnotation()
        
    else:
      print("Check UUID ({0}), no such image on dataset.".format(uuid))

  def updateImageBase(self):
    """
    Revisits image base directory and add any new images to the cache.
    """
    wsilist = os.listdir(os.path.join(self.path,"WSI"))

    for f in wsilist:
      if not f in self._wsi:
        self.addImageFile(f)

  
  def searchBCR(self,tcga):
    """
    Search for a BCR file that corresponds to a new image file.
    """

    bcrlist = os.listdir(os.path.join(self.path,"BCR"))

    for f in bcrlist:
      case_files = os.listdir(os.path.join(self.path,"BCR",f))
      for item in case_files:
        if BCRParser.checkFileName(item):
          bcr_obj = BCRParser(os.path.join(self.path,"BCR",f,item))
          key = bcr_obj.returnPatientIDFromFile()
          if key == tcga.returnCaseID():
            return [bcr_obj]
          else:
            return []
        
  def returnNumberOfCases(self):
    """
    Self explanatory.
    """
    return len(self._data)
  
  def findGBMCases(self):
    """
    Populates a dictionary with Case IDs from GBM patients.
    Dictionary has the same structure as main data.
    
    To see GBM cases use functions:
    returnGBMCasesWithImages(self)
    returnGBMImage(patientID)
    """

    if self._gbm is None:
      self._gbm = dict()
      
    for key in self._data:
      bcrlist = self._data[key]['bcr']
      if len(bcrlist) == 1:
        bcr = bcrlist[0]
        if bcr.returnDiseaseCode() == 'GBM':
          self._gbm[key] = self._data[key]
      else:
        for bcr in bcrlist:
          print("Multiple BCR reports for a single patient. Check BCR UUID: {0}".format(bcr.returnDirectoryUUID()))
          
    return len(self._gbm)

  def returnGBMCasesWithImages(self):
    """
    Return a list of GBM patient IDs cases for which slide images are available.
    """
    print("To be implemented.")


  def returnGBMImage(self,patientID):
    """
    Return a BCR object for a GBM case.
    """
    print("To be implemented.")
    
  
  def findLGGCases(self):
    """
    Populates a dictionary with all LGG cases.
    The dictionary has the same layout as the _data.

    To see LGG cases use functions:
    returnLGGCasesWithImages(self)
    returnLGGImage(patientID)
    """
    if self._lgg is None:
      self._lgg = dict()
      
    for key in self._data:
      bcrlist = self._data[key]['bcr']
      if len(bcrlist) == 1:
        bcr = bcrlist[0]
        if bcr.returnDiseaseCode() == 'LGG':
          self._lgg[key] = self._data[key]
      else:
        for bcr in bcrlist:
          print("Multiple BCR reports for a single patient. Check BCR UUID: {0}".format(bcr.returnDirectoryUUID()))

    return len(self._lgg)

  def returnGBMPrimary(self):
    """
    Identify GBM primary (de novo) and Treated primary GBMs. The first type corresponds to a GBM that a primary occurance whereas the second 
    corresponds to a lower grade glioma that evolved into a GBM.
    
    Returns a dictionary with all cases of each type.
    """

    if self._gbm is None:
      self.findGBMCases()

    primary = dict()
    for case in self._gbm:
      bcrlist = self._data[case]['bcr']
      if len(bcrlist) == 1:
        case_type = bcrlist[0].returnHistologicalType()
        if case_type in primary:
          primary[case_type].append(case)
        else:
          primary[case_type] = [case]
      else:
        print("Multiple BCR reports for a single patient. Check BCR UUID: {0}".format(bcrlist[0].returnDirectoryUUID()))

    return primary
      
  def returnLGGSubtypes(self):
    """
    Return a dictionary containing each LGG subtype as a key and the values are lists of CaseIDs corresponding to the subtype.
    """
    if self._lgg is None:
      self.findLGGCases()

    subtypes = dict()
    for case in self._lgg:
      bcrlist = self._lgg[case]['bcr']
      if len(bcrlist) == 1:
        case_type = bcrlist[0].returnHistologicalType()
        if case_type in subtypes:
          subtypes[case_type].append(case)
        else:
          subtypes[case_type] = [case]
      else:
        for bcr in bcrlist:
          print("Multiple BCR reports for a single patient. Check BCR UUID: {0}".format(bcr.returnDirectoryUUID()))
    return subtypes
  
  def returnMeanSlidesPerCase(self):
    """
    Returns the average number of slides in each case.
    """
    return len(self._wsi)/len(self._data)

  def returnMaximumSlidesInCase(self):
    """
    Find and returns a tuple with 2 values:
    1 - list of cases with the most slides;
    2 - number of slides in case
    """
    maxCase = []
    maxNumber = 1
    
    for key in self._data:
      check = len(self._data[key]['wsi'])
      if check >= maxNumber:
        maxNumber = check
        maxCase.append(key)

    return (maxCase,maxNumber)

  def findAnnotations(self):
    """
    Returns a list of WSI directory UUIDs which contains annotations.
    """
    tcga_list = []
    for key in self._wsi:
      if self._wsi[key].returnAnnotation():
        tcga_list.append(key)
    return tcga_list
  
  def returnCaseFromSlide(self,uuid):
    """
    uuid: is the image's directory uuid
    Return case data that a slide is a part of.
    Returned object is a BCR instance.
    """
    if uuid in self._wsi:
      tcga = self._wsi[uuid]
      bcrlist = self._data[tcga.returnCaseID()]['bcr']
      if len(bcrlist) == 1:
        return bcrlist[0]
      else:
        for bcr in bcrlist:
          print("Multiple BCR reports for a single patient. Check BCR UUID: {0}".format(bcr.returnDirectoryUUID()))
        return None
    else:
      print("Image UUID not on record.")
      return None
  
  def returnCaseDataFromID(self,caseID):
    """
    Return case BCR data from ID.
    """
    bcrdata = self._data[caseID]['bcr']

    if len(bcrdata) > 1:
      for bcr in bcrdata:
        print("Multiple BCR reports for a single patient. Check BCR UUID: {0}".format(bcr.returnDirectoryUUID()))
        return None
    else:
      return bcrdata[0]

  def returnImageFromUUID(self,uuid):
    """
    Returns the TCGA Object for the image in directory UUID.
    """
    if uuid in self._wsi:
      return self._wsi[uuid]
    else:
      print("No such directory UUID {0}".format(uuid))
      return None
    
  def returnImagesInCase(self,caseID):
    """
    Return the image list associated to a case.
    """
    return self._data[caseID]['wsi']

  def returnSlideTypes(self):
    """
    Returns the number of slides of each glioma type or subtype.
    """
    subtypes = dict()
    
    for uuid in self._wsi:
      bcr = self.returnCaseFromSlide(uuid)
      case_type = bcr.returnHistologicalType()
      if case_type in subtypes:
        subtypes[case_type].append(bcr.returnPatientIDFromFile())
      else:
        subtypes[case_type] = [bcr.returnPatientIDFromFile()]

    return subtypes
  
  def storeState(self):
    """
    Dump dictionary before deletion.
    """
    if self._pickfd is None or self._pickfd.closed:
      self._pickfd = open(os.path.join(self.path,self._pickFile),"wb")

    if self._wsifd is None or self._wsifd.closed:
      self._wsifd = open(os.path.join(self.path,self._wsiFile),"wb")

    print("Starting to dump WSI data.")
    pickle.dump(self._wsi,self._wsifd,-1)
    
    print("Starting to dump BCR data.")
    pickle.dump(self._data,self._pickfd,-1)

    self._pickfd.close()
    self._wsifd.close()


def checkForDuplicateData(merger,cases):
    images = dict()
    
    for caseID in cases:
        for image in merger.returnImagesInCase(caseID):
            uuid = image.returnUUID()
            if uuid in images:
                images[uuid].append(caseID)
            else:
                images[uuid] = [caseID]
    return images

if __name__ == "__main__":

    print("Unit test")
    dataset = "/Volumes/Trabalho/Doutorado/Dataset/"
    uuid = "0728b332-782a-41df-b863-a0751dc4740d"
    iuuid = "a6c4ff52-42bc-4f1b-b63f-63c69488d5d7"
    
    m = Merger(dataset)
    #Test phase, search for data, generate statistics
    #Test image addition
    #uuid = input("Enter UUID for new image or hit enter to continue: ")
    print("Image add test.")
    if uuid:
        m.addImageFile(uuid)

    #Number of cases
    print("Number of cases {0:d}".format(m.returnNumberOfCases()))
    
    #Find GBM cases
    print("Total of GBM cases: {0:d}".format(m.findGBMCases()))
    
    #Find LGG cases
    print("Total of LGG cases: {0:d}".format(m.findLGGCases()))
    
    #Case with most number of slides
    cases = m.returnMaximumSlidesInCase()
    print("A total of {0:d} cases have most number of slides ({1}). Cases are: \n {2}".format(len(cases[0]),cases[1],cases[0]))
    
    #Inspect cases
    #for case in cases[0]:
    #    print("Case BCR UUID: {0}".format(m.returnCaseData(case).returnDirectoryUUID()))
    #Check Duplicat Images in Cases
    #images = checkForDuplicateData(m,cases[0])
    #for image in images:
    #    print("Image {0} is related to case(s): {1}".format(image,images[image]))

    #Retrieve BCR data corresponding to an image UUID
    print("BCR data corresponding to image directory {0}: patient ID {1}".format(iuuid,m.returnCaseFromSlide(iuuid).returnPatientIDFromXML()))

    #Test update image base
    print("Updating WSI base.")
    m.updateImageBase()

    #Retrieve slides with annotations
    slides = m.findAnnotations()
    if not slides is None:
        for wsi in slides:
            print("Slide in directory UUID {0}, regarding patient {1} has annotations".format(wsi,m.returnImageFromUUID(wsi).returnCaseID()))
    else:
        print("No annotations found.")

    #Retrieve subtypes
    subtypes = m.returnLGGSubtypes()
    for subtype in subtypes:
        print("LGG subtype {0} has {1:d} cases.".format(subtype,len(subtypes[subtype])))

    #Images available by subtype
    imagetypes = m.returnSlideTypes()
    for subtype in imagetypes:
        print("There are {0:d} images of subtype {1}".format(len(imagetypes[subtype]),subtype))

    #GBM case origin
    gbm_cases = m.returnGBMPrimary()
    for gbm_type in gbm_cases:
        print("There are {0:d} cases of {1} in repository.".format(len(gbm_cases[gbm_type]),gbm_type))
        
    #Terminate
    m.storeState()
    

