@echo off
REM prepare_training.bat
REM author: Julie Kallini

setlocal

set "MISTRAL_PATH=C:\Users\abathur\Desktop\MscProject\mistral"

echo.
echo ------------------------------------------------------------------------------
echo Arguments
echo ------------------------------------------------------------------------------
echo Perturbation type: %1
echo Train set: %2
echo Random seed: %3
echo Parent pretrained model: %4

set "NO_POS_ENCODINGS=%5"
if "%NO_POS_ENCODINGS%"=="" (
    set "NPS="
    set "NPSunderscore="
) else (
    set "NPS=-no-positional-encodings"
    set "NPSunderscore=_no_positional_encodings"
)

echo No pos encodings: %NO_POS_ENCODINGS%
echo Mistral path: %MISTRAL_PATH%

echo.
echo ------------------------------------------------------------------------------
echo Generating yaml files for mistral training
echo ------------------------------------------------------------------------------
echo python generate_yaml.py %1 %2 %3 %4 %NO_POS_ENCODINGS%
python generate_yaml.py %1 %2 %3 %4 %NO_POS_ENCODINGS%

echo.
echo ------------------------------------------------------------------------------
echo Copying config yaml files to mistral directory
echo ------------------------------------------------------------------------------
set "SRC1=conf\babylm_%1_%2_%4%NPSunderscore%\seed%3\dataset_%1_%2_seed%3.yaml"
set "DST1=%MISTRAL_PATH%\conf\datasets\dataset_%1_%2_seed%3.yaml"
echo copy "%SRC1%" "%DST1%"
copy "%SRC1%" "%DST1%"

set "SRC2=conf\babylm_%1_%2_%4%NPSunderscore%\seed%3\train_%1_%2_%4%NPSunderscore%_seed%3.yaml"
set "DST2=%MISTRAL_PATH%\conf\train_%1_%2_%4%NPSunderscore%_seed%3.yaml"
echo copy "%SRC2%" "%DST2%"
copy "%SRC2%" "%DST2%"

set "SRC3=conf\babylm_%1_%2_%4%NPSunderscore%\gpt2%NPS%-small-%1-%4.yaml"
set "DST3=%MISTRAL_PATH%\conf\models\gpt2%NPS%-small-%1-%4.yaml"
echo copy "%SRC3%" "%DST3%"
copy "%SRC3%" "%DST3%"

echo.
echo ------------------------------------------------------------------------------
echo Done!
echo ------------------------------------------------------------------------------
endlocal
