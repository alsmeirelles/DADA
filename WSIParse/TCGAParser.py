#!/usr/bin/env python3
#-*- coding: utf-8

import os,sys
import re

class TCGABarcode(object):
  """
  This class parses TCGA barcode used to name files within TCGA.
  Reference: https://wiki.nci.nih.gov/display/TCGA/TCGA+barcode
  
  Expected filenames are like: TCGA-02-0321-01A-01-BS1.c8aacb16-a44d-457f-95ac-40f9cd146a2f.svs
  
  First part, separeted by a dot is the barcode. Second part is a UUID. Third part is file extension.

  @param code <str>: SVS file code
  @param dir_path <str>: path to directory containing image
  """

  _rex = r'(?P<tcga>TCGA)-(?P<tss>[\w]{2})-(?P<part>[\w]{4})-(?P<sample>[\d]{2}[A-Z]{0,1})-(?P<portion>[\d]{2}[A-Z]{0,1})-(?P<plate>[\w]{4}){0,1}(?P<tissue>(TS|BS|MS)\w)'
  BARCODE = 0
  UUID = 1
      
  def __init__(self,code,dir_path,verbose=0):
    self.matchP = None
    self._verbose = verbose
    
    if isinstance(code,str):
      self.code = code.split(".")
    else:
      raise ValueError("Wrong argument type for TCGABarcode instance, should be string")

    self.recomp = re.compile(self._rex)
    self.matchP = self.recomp.match(self.code[self.BARCODE])

    if not self.matchP:
      raise ValueError("File name does not match pattern: {0}.".format(code))

    self._dirUUID = os.path.basename(dir_path)
    self._path = os.path.join(dir_path,code)
    self._hasAnnotation = False

  def getPath(self):
    return self._path
  
  def setAnnotation(self):
    """
    Set if slide is accompanied by annotations.
    """
    self._hasAnnotation = True

  def returnAnnotation(self):
    """
    Return status about annotation presence.
    """
    return self._hasAnnotation

  def returnDirectoryUUID(self):
    """
    Returns the directory's uuid for the slide.
    """
    return self._dirUUID
  
  def returnTss(self):
    """
    Return the TSS field from the given barcode.

    Code values are listed in: https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tissue-source-site-codes
    """
    return self.matchP.group('tss')

  def returnParticipant(self):
    """
    Return the participant field from the given barcode.
    """
    return self.matchP.group('part')
  
  def returnSample(self):
    """
    Return the sample field from the given barcode.
    """
    return self.matchP.group('sample')

  def returnPortion(self):
    """
    Return the portion field from the given barcode.
    """
    return self.matchP.group('portion')

  def returnPlate(self):
    """
    Return the plate field from the given barcode, if it exists.
    """
    return self.matchP.group('plate')

  def returnTissue(self):
    """
    Return the tissue field from the given barcode.
    
    TS - Top slide
    MS - Middle slide
    BS - Bottom slide
    """
    return self.matchP.group('tissue')

  def returnUUID(self):
    """
    Return file UUID
    """
    return self.code[self.UUID]

  def returnCaseID(self):
    """
    Return the case ID, in accordance with the barcode convention.
    """
    return "-".join(self.matchP.group(1,2,3))

  def __setstate__(self,state):
      """
      Recovers match objects.
      """
      recomp = re.compile(self._rex)
      code = state['code']
      state['recomp'] = recomp
      state['matchP'] = recomp.match(code[self.BARCODE])
      self.__dict__.update(state)
      
  def __getstate__(self):
      """
      Prepares for pickling.
      """
      state = self.__dict__.copy()
      del state['matchP']
      del state['recomp']

      return state

  @classmethod
  def checkFileName(cls,fname):
    """
    Checks a file name for a pattern match.
    """
    comp = re.compile(cls._rex)
    code = fname.split('.')[cls.BARCODE]
    if comp.match(code):
        return True
    else:
        return False

if __name__ == "__main__":
    """
    This main code should be used for testing purposes only.
    """

    #Unit testing
    filename = "TCGA-DU-7007-01A-01-TS1.b6621bdd-067f-4334-a452-935e39adccf4.svs"
    uuid = '/Volumes/Trabalho/Doutorado/Dataset/WSI/a268ed54-3633-4597-82ea-3e4107a77d77'
    t = TCGABarcode(filename,uuid)

    if not t:
        print("Not this time...")
        sys.exit(-1)

    output = "{0}:{1}"

    print(output.format("tss",t.returnTss()))
    print(output.format("participant",t.returnParticipant()))
    print(output.format("sample",t.returnSample()))
    print(output.format("portion",t.returnPortion()))
    print(output.format("plate",t.returnPlate()))
    print(output.format("tissue",t.returnTissue()))
    print(output.format("UUID",t.returnUUID()))
    print(output.format("Case ID",t.returnCaseID()))

    if TCGABarcode.checkFileName(filename):
        print("File name check.")

