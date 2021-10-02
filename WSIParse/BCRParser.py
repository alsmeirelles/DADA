#!/usr/bin/env python3
#-*- coding: utf-8

try:
  from lxml import etree
  print("running with lxml.etree")
except ImportError:
  try:
    # Python 2.5
    import xml.etree.cElementTree as etree
    print("running with cElementTree on Python 2.5+")
  except ImportError:
    try:
      # Python 2.5
      import xml.etree.ElementTree as etree
      print("running with ElementTree on Python 2.5+")
    except ImportError:
      try:
        # normal cElementTree install
        import cElementTree as etree
        print("running with cElementTree")
      except ImportError:
        try:
          # normal ElementTree install
          import elementtree.ElementTree as etree
          print("running with ElementTree")
        except ImportError:
          print("Failed to import ElementTree from any known place")


import os
import re


class BCRParser(object):
  """
  Parse and return tags and values from a BCR file.

  BCR files consist of 2 elements:
  -admin: administrative basic data
  -patient: patient specific data
  """

  _rex = r'(?P<hosp>[\w]+).org_clinical.(?P<tcga>TCGA)-(?P<tss>[\w]{2})-(?P<part>[\w]{4}).xml'

  #Common requested fields
  _DISEASE_CODE = 'disease_code'
  _HISTOLOGICAL_TYPE = 'histological_type'
  _PATIENT_UUID = 'bcr_patient_uuid'
  _PATIENT_ID = 'bcr_patient_barcode'

  _nsmap = {'lgg':"http://tcga.nci/bcr/xml/clinical/lgg/2.7",
            'xsi':"http://www.w3.org/2001/XMLSchema-instance",
            'admin':"http://tcga.nci/bcr/xml/administration/2.7",
            'clin_shared':"http://tcga.nci/bcr/xml/clinical/shared/2.7",
            'shared':"http://tcga.nci/bcr/xml/shared/2.7",
            'lgg_nte':"http://tcga.nci/bcr/xml/clinical/lgg/shared/new_tumor_event/2.7/1.0",
            'nte':"http://tcga.nci/bcr/xml/clinical/shared/new_tumor_event/2.7",
            'rx':"http://tcga.nci/bcr/xml/clinical/pharmaceutical/2.7",
            'rad':"http://tcga.nci/bcr/xml/clinical/radiation/2.7",
            'follow_up_v1.0':"http://tcga.nci/bcr/xml/clinical/lgg/followup/2.7/1.0"}

  def __init__(self,filename):

      if not (os.path.isfile(filename) and filename.endswith("xml")):
          raise ValueError("File not found: {0}".format(filename))
          
      self.filename = filename
      self._cachedData = {}
      
      fd = open(self.filename,"rb")
      self._xmlParser = etree.parse(fd)
      self._xmlRoot = self._xmlParser.getroot()
      fd.close()

  def returnPatientIDFromFile(self):
      """
      Patient IDs are inserted into BCR file names. Return those IDs.
      """
      basename = os.path.basename(self.filename)
      return basename.split(".")[2]

  def returnDiseaseCode(self):
      """
      Disease codes for gliomas are:
      LGG or GBM
      """
      if self._DISEASE_CODE in self._cachedData:
          return self._cachedData[self._DISEASE_CODE]
      
      element = self.xmlFindElement('admin',self._DISEASE_CODE)
      if element is not None:
          self._cachedData[self._DISEASE_CODE] = element.text
          return element.text
      else:
          print("Element not found in file.")

  def returnHistologicalType(self):
      """
      Return glioma subtype.
      """
      if self._HISTOLOGICAL_TYPE in self._cachedData:
          return self._cachedData[self._HISTOLOGICAL_TYPE]
      
      element = self.xmlFindElement('shared',self._HISTOLOGICAL_TYPE)
      if element is not None:
          self._cachedData[self._HISTOLOGICAL_TYPE] = element.text
          return element.text
      else:
          print("Element not found in file.")

  def returnPatientUUID(self):
      """
      Return patiend UUID.
      """
      if self._PATIENT_UUID in self._cachedData:
          return self._cachedData[self._PATIENT_UUID]
      
      element = self.xmlFindElement('shared',self._PATIENT_UUID)
      if element is not None:
          self._cachedData[self._PATIENT_UUID] = element.text
          return element.text
      else:
          print("Element not found in file.")

  def returnPatientIDFromXML(self):
      """
      Return patiend barcode ID (Case ID).
      """
      if self._PATIENT_ID in self._cachedData:
          return self._cachedData[self._PATIENT_ID]
      
      element = self.xmlFindElement('shared',self._PATIENT_ID)
      if element is not None:
          self._cachedData[self._PATIENT_ID] = element.text
          return element.text
      else:
          print("Element not found in file.")

  def xmlFindElement(self,namespace,elementName):
      """
      Return the XML element corresponding to specified parameters.
      """

      if namespace == 'admin':
          elementIndex = 0
      else:
          elementIndex = 1

      if self._xmlParser is None or self._xmlRoot is None:
          fd = open(self.filename,"rb")          
          self._xmlParser = etree.parse(fd)
          self._xmlRoot = self._xmlParser.getroot()
          fd.close()

      try:
        xml_namespace = self._xmlRoot[elementIndex].nsmap[namespace]
        element = self._xmlRoot[elementIndex].find("{{{0}}}{1}".format(xml_namespace,elementName))
      except AttributeError:
        element = self._xmlRoot[elementIndex].find('{{http://tcga.nci/bcr/xml/administration/2.7}}{0}'.format(elementName))

      return element

  def returnDirectoryUUID(self):
      """
      Returns the directory UUID in which the BCR xml is in.
      """
      return os.path.split(os.path.dirname(self.filename))[1]

  def flushElementTree(self):
      """
      Etree objects can not be pickled. Flush lxml objects to pickle.
      """
      
      del(self._xmlParser)
      del(self._xmlRoot)

      self._xmlParser = None
      self._xmlRoot = None

  def __getstate__(self):
      """
      lxml etree objects can't be pickled
      """

      state = self.__dict__.copy()
      if '_xmlParser' in state:
        del state['_xmlParser']

      if '_xmlRoot' in state:
        del state['_xmlRoot']

      return state

#  def __reduce__(self):
#      """
#      What to recreate during unpickling.
#      """
#      return (self.__class__,(),(self.filename, self._cachedData))

  def __setstate__(self,state):
      """
      Recover previous state.
      """
      
      fd = open(state['filename'],"rb")
      _xmlParser = etree.parse(fd)
      _xmlRoot = _xmlParser.getroot()
      state['_xmlParser'] = _xmlParser
      state['_xmlRoot'] = _xmlRoot
      fd.close()
      
      self.__dict__.update(state)
      
  @classmethod
  def checkFileName(cls,fname):
      """
      Checks a file name for a pattern match.
      """
      comp = re.compile(cls._rex)
      if comp.match(fname):
          return True
      else:
          return False

if __name__ == "__main__":
    
    filename = "../Dataset/BCR/28e551c3-fb78-4505-9376-ee9b5121ee58/nationwidechildrens.org_clinical.TCGA-06-6700.xml"
    bcr = BCRParser(filename)
    output = "{0}: {1}"

    print(output.format("Patient ID from file name",bcr.returnPatientIDFromFile()))
    print(output.format("Disease code",bcr.returnDiseaseCode()))
    print(output.format("Histological type",bcr.returnHistologicalType()))
    print(output.format("Patient UUID",bcr.returnPatientUUID()))
    print(output.format("Patient ID from XML",bcr.returnPatientIDFromXML()))
    print(output.format("Directory UUID from filename",bcr.returnDirectoryUUID()))
    if checkFileName(os.path.basename(filename)):
        print("File name check.")

    print("Flushing lxml objects")
    bcr.flushElementTree()
    if bcr._xmlParser is None and bcr._xmlRoot is None:
        print("Flushsed!")

    print("Parsing again:")
    print(output.format("Patient UUID",bcr.returnPatientUUID()))
