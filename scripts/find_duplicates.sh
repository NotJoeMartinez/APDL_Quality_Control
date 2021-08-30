# Shell script for finding duplicate images

get_sha_sums () { 


    for D in *; do
        if [ -d "${D}" ]; then
            echo "${D}"   
            # get sha sums from directory then save them to a sorted foo.txt file 
            identify -format "%# %f\n" ${D}/*.jpg| sort >> foo.txt

        fi
    done


}

find_duplicates () {
        # save output of shasums to a foo.txt 
    MATCHES=$(grep -oh -E "^(.{64})\s" foo.txt | uniq -d)

    # save matches to matches.txt
    echo "${MATCHES}" > matches.txt

    # reads matches.txt and looks for finds in foo.txt 
    while read p; do
        grep "$p" foo.txt >> yeet.txt
    done <matches.txt 



    while read i; do
        # FOO=$(grep -oh -E "\s(.{32}\.jpg)" $i)
        FOO=$(echo $i | grep -oh -E "\s(.{32}\.jpg)")
        find . -type f -name $FOO >> paths.txt
    done <yeet.txt 

    rm yeet.txt


}

get_sha_sums
find_duplicates 