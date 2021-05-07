# Script to rename images and convert them to pngs

mkdir /tmp/renamed_imgs

for D in *; do
    if [ -d "${D}" ]; then
        echo "${D}"   
		cd ${D} 
		rm *.txt # remove all text files
		mogrify -format jpg *.png # convert all pngs to jpg
		rm *.png # remove all png
		for i in *.jpg; do mv -i "$i" $(keepassxc-cli generate).jpg; done; # rename all jps
		cd ..
		cp ${D}/* /tmp/renamed_imgs/ # move all renamed images into un_classifed
    fi
done

mv /tmp/renamed_imgs .
