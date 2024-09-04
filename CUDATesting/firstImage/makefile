
all: win
# all: linux

win: firstImage.cu
	nvcc -o firstImage firstImage.cu

linux: firstImage.cu # will convert to use g++ instead of cl.exe
	nvcc -o firstImage firstImage.cu

win-run: ./firstImage.exe
	./firstImage.exe | Out-File -FilePath img.ppm -Encoding ASCII

linux-run: ./firstImage #default encoding is ASCII
	./firstImage > img.ppm 