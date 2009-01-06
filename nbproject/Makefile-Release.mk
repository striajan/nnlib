#
# Gererated Makefile - do not edit!
#
# Edit the Makefile in the project folder instead (../Makefile). Each target
# has a -pre and a -post target defined where you can add customized code.
#
# This makefile implements configuration specific macros and targets.


# Environment
MKDIR=mkdir
CP=cp
CCADMIN=CCadmin
RANLIB=ranlib
CC=gcc
CCC=g++
CXX=g++
FC=

# Include project Makefile
include Makefile

# Object Directory
OBJECTDIR=build/Release/GNU-Linux-x86

# Object Files
OBJECTFILES= \
	${OBJECTDIR}/_ext/home/honza/data/Dokumenty/Neuronove_site/nnlib/src/backPropagation/continuator.o \
	${OBJECTDIR}/src/main.o

# C Compiler Flags
CFLAGS=

# CC Compiler Flags
CCFLAGS=-msse
CXXFLAGS=-msse

# Fortran Compiler Flags
FFLAGS=

# Link Libraries and Options
LDLIBSOPTIONS=

# Build Targets
.build-conf: ${BUILD_SUBPROJECTS} out/Release/GNU-Linux-x86

out/Release/GNU-Linux-x86: ${OBJECTFILES}
	${MKDIR} -p out/Release
	${LINK.cc} -o out/Release/GNU-Linux-x86 ${OBJECTFILES} ${LDLIBSOPTIONS} 

${OBJECTDIR}/_ext/home/honza/data/Dokumenty/Neuronove_site/nnlib/src/backPropagation/continuator.o: /home/honza/data/Dokumenty/Neuronove\ site/nnlib/src/backPropagation/continuator.cpp 
	${MKDIR} -p ${OBJECTDIR}/_ext/home/honza/data/Dokumenty/Neuronove_site/nnlib/src/backPropagation
	$(COMPILE.cc) -O3 -Wall -Isrc -o ${OBJECTDIR}/_ext/home/honza/data/Dokumenty/Neuronove_site/nnlib/src/backPropagation/continuator.o /home/honza/data/Dokumenty/Neuronove\ site/nnlib/src/backPropagation/continuator.cpp

${OBJECTDIR}/src/main.o: src/main.cpp 
	${MKDIR} -p ${OBJECTDIR}/src
	$(COMPILE.cc) -O3 -Wall -Isrc -o ${OBJECTDIR}/src/main.o src/main.cpp

# Subprojects
.build-subprojects:

# Clean Targets
.clean-conf:
	${RM} -r build/Release
	${RM} out/Release/GNU-Linux-x86

# Subprojects
.clean-subprojects:
