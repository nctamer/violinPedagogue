### GRADE 1 ###
yt-dlp  https://www.youtube.com/watch?v=JTzSxS3q9Zs  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Suzuki Violin School Vol. 1"  -o "chapter:L1/S1_BochanKang_%(section_number)03d_%(section_title)s.%(ext)s"
yt-dlp https://www.youtube.com/watch?v=6Y8yHaKTKuw  -x --audio-format mp3 --audio-quality 0 --split-chapters -o "chapter:L1/S1_YeraLee_%(section_number)03d_%(section_title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 5-26 https://youtube.com/playlist?list=PLMFLV1svOUsQCTTk8a1tAZG0BYr2CrF4f -x --audio-format mp3 --audio-quality 0   -o "L1/S1_MyriadMSDA_%(playlist_index)03d_%(title)s.%(ext)s"
rm \[Suzuki\ Violin\ School\ \ Book\ 1\]\ ALL\ Songs\,\ 스즈키\ 바이올린\ 1권\ 전곡\ 수록\ \[6Y8yHaKTKuw\].mp3
rm 'Suzuki Violin School Book Vol. 1 Full Version  @보찬TV<200b> [JTzSxS3q9Zs].mp3'


### GRADE 2 ###
yt-dlp  https://www.youtube.com/watch?v=K6NV1vZhnqU  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Suzuki Violin School Vol. 2" --remove-chapters "Mignon"  -o "chapter:L2/S2_BochanKang_%(section_number)03d_%(section_title)s.%(ext)s"
yt-dlp https://www.youtube.com/watch?v=M9s6rEct07g  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Mignon" -o "chapter:L2/S2_YeraLee_%(section_number)03d_%(section_title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-8,10-12 https://youtube.com/playlist?list=PLMFLV1svOUsTEGLJEh7Yt4GSmzColeSY7 -x --audio-format mp3 --audio-quality 0   -o "L2/S2_MyriadMSDA_%(playlist_index)03d_%(title)s.%(ext)s"
rm Suzuki\ Violin\ School\ Book\ Vol.\ 2\ Full\ Version\ @보찬TV\ \[K6NV1vZhnqU\].mp3
rm \[Suzuki\ Violin\ Book\ 2\]\ All\ Songs\,\ 스즈키\ 바이올린\ 2권\ \(전곡\ 수록\)\ \[M9s6rEct07g\].mp3


### GRADE 3 ###
yt-dlp  https://www.youtube.com/watch?v=IgnEw3E4_L8  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Suzuki Violin School" --remove-chapters "6.Gavotte"  --remove-chapters "7.Bourrée"  -o "chapter:L3/S3_BochanKang_%(section_number)03d_%(section_title)s.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=n30OtrMWkrE  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "3-6 Gavotte in DM" --remove-chapters "3-7 Bourree" -o "chapter:L3/S3_YeraLee_%(section_number)03d_%(section_title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-8,10-12 https://youtube.com/playlist?list=PLMFLV1svOUsSEHvPoY64L1machR4j1mrN -x --audio-format mp3 --audio-quality 0   -o "L3/S2_MyriadMSDA_%(playlist_index)03d_%(title)s.%(ext)s"
rm Suzuki\ Violin\ School\ Book\ Vol.\ 3\ Full\ Version\ @보찬TV\ \[IgnEw3E4_L8\].mp3 
rm \[Suzuki\ Violin\ School\ Book\ 3\]\ ALL\ Songs\,\ 스즈키\ 바이올린\ 3권\ 전곡\ 수록\ \[n30OtrMWkrE\].mp3
yt-dlp --yes-playlist --playlist-items 1-12,15-17 https://www.youtube.com/playlist?list=OLAK5uy_mESjCL-XUZUHoO2T_uXGd-q4YYJhgbLOo -x --audio-format mp3 --audio-quality 0   -o "L3/Wohlfahrt45_JPRafferty_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-12,15-17 https://youtube.com/playlist?list=PLQT2_mTTuV128kC6hL-W2aw5Ntj7Ed3mO -x --audio-format mp3 --audio-quality 0   -o "L3/Wohlfahrt45_BernardChevalier_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-12,15-17 https://youtube.com/playlist?list=PLatxbbcShnHWgHACAchZ_EGyZmMAi73mP -x --audio-format mp3 --audio-quality 0   -o "L3/Wohlfahrt45_TimRohwer_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-12,15-17 https://youtube.com/playlist?list=PLCA313AAE75E6FEED -x --audio-format mp3 --audio-quality 0   -o "L3/Wohlfahrt45_BrianClement_%(playlist_index)03d_%(title)s.%(ext)s"



### GRADE 4 ###
yt-dlp --yes-playlist --playlist-items 18-20,22-26,31,32,34-37 https://www.youtube.com/playlist?list=OLAK5uy_mESjCL-XUZUHoO2T_uXGd-q4YYJhgbLOo -x --audio-format mp3 --audio-quality 0   -o "L4/Wohlfahrt45_JPRafferty_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 18-20,22-26,31,32,34-37  https://youtube.com/playlist?list=PLQT2_mTTuV128kC6hL-W2aw5Ntj7Ed3mO -x --audio-format mp3 --audio-quality 0   -o "L4/Wohlfahrt45_BernardChevalier_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 18-20,22-26,31,32,34-37  https://youtube.com/playlist?list=PLatxbbcShnHWgHACAchZ_EGyZmMAi73mP -x --audio-format mp3 --audio-quality 0   -o "L4/Wohlfahrt45_TimRohwer_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 18-20,22-26,31,32,34-37  https://youtube.com/playlist?list=PLCA313AAE75E6FEED -x --audio-format mp3 --audio-quality 0   -o "L4/Wohlfahrt45_BrianClement_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 4,5,7 --postprocessor-args "ffmpeg:-ss 0:0:5" https://youtube.com/playlist?list=PLHyVuM6mAj71QYqb24xkVwUR_0qTfg228 -x --audio-format mp3 --audio-quality 0   -o "L4/S4_MikeChau_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 2,6,7,9 --postprocessor-args "ffmpeg:-ss 0:0:5" https://youtube.com/playlist?list=PLHyVuM6mAj72ZUNMbEF3ITNf7r2gAsbYu -x --audio-format mp3 --audio-quality 0   -o "L4/S4_MikeChau_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=Rz5_5hFlSJ8  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Suzuki Violin School" --remove-chapters "F. Seitz"  -o "chapter:L4/S4_BochanKang_%(section_number)03d_%(section_title)s.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=TU9J9g783XE  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Suzuki Violin School" --remove-chapters "Gavotte" --remove-chapters "Concerto in G Minor" --remove-chapters "F. M. Veracini" -o "chapter:L4/S5_BochanKang_%(section_number)03d_%(section_title)s.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=ttHaG9vM5O8  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Seitz" --remove-chapters "Perpetual Motion" -o "chapter:L4/S4_YeraLee_%(section_number)03d_%(section_title)s.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=--0L9xoZJb0  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Gavotte" --remove-chapters "G minor" --remove-chapters "Gigue" -o "chapter:L4/S5_YeraLee_%(section_number)03d_%(section_title)s.%(ext)s"
rm '[Suzuki Violin Book 4] All Songs, 스즈키 바이올린 4권 (전곡 듣기) [ttHaG9vM5O8].mp3'
rm 'Suzuki Violin Book 5, All Songs, 스즈키 바이올린 5권, 모든곡 [--0L9xoZJb0].mp3'
rm 'Suzuki Violin School Book Vol. 4 Full Version @보찬TV [Rz5_5hFlSJ8].mp3'
rm 'Suzuki Violin School Book Vol. 5 Full Version @보찬TV [TU9J9g783XE].mp3'
yt-dlp --yes-playlist --playlist-items 2-5 https://youtube.com/playlist?list=OLAK5uy_njnpf7e-zqO6YhbCucfU5SDXOuqFUz8F8 -x --audio-format mp3 --audio-quality 0   -o "L4/Kreutzer_SunKim_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 2-5 https://youtube.com/playlist?list=PL57E3A778DD644CC5 -x --audio-format mp3 --audio-quality 0   -o "L4/Kreutzer_BernardChevalier_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-4 https://youtube.com/playlist?list=PLIcJOrQKnxs0-_OhkyymPTmzJktkjkNuA -x --audio-format mp3 --audio-quality 0   -o "L4/Kreutzer_BochanKang_%(playlist_index)03d_%(title)s.%(ext)s"




### GRADE 5 ###
yt-dlp --yes-playlist --playlist-items 39,41-42,45,48-52,57,58 https://www.youtube.com/playlist?list=OLAK5uy_mESjCL-XUZUHoO2T_uXGd-q4YYJhgbLOo -x --audio-format mp3 --audio-quality 0   -o "L5/Wohlfahrt45_JPRafferty_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 39,41-42,45,48-52,57,58 https://youtube.com/playlist?list=PLQT2_mTTuV128kC6hL-W2aw5Ntj7Ed3mO -x --audio-format mp3 --audio-quality 0   -o "L5/Wohlfahrt45_BernardChevalier_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 39,41-42,45,48-52,57,58  https://youtube.com/playlist?list=PLCA313AAE75E6FEED -x --audio-format mp3 --audio-quality 0   -o "L5/Wohlfahrt45_BrianClement_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1,2,10,12,30  --postprocessor-args "ffmpeg:-ss 0:0:3" https://youtube.com/playlist?list=PLIcJOrQKnxs2L67LkeccL8K9L_Uv8raUo -x --audio-format mp3 --audio-quality 0   -o "L5/Kayser20_BochanKang_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1,2,10,12,30 --postprocessor-args "ffmpeg:-ss 0:0:4" https://youtube.com/playlist?list=PLTdeeeVSPtA0-ZdTT--hw4-R8FSG82AOS -x --audio-format mp3 --audio-quality 0   -o "L5/Kayser20_AlexandrosIakovou_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1,2,10,12,30  https://youtube.com/playlist?list=PLQyxywuv2Kp40hffFLWpqypWP-XHOaiho -x --audio-format mp3 --audio-quality 0   -o "L5/Kayser20_FabricioValvasori_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 8,10,12,13,20,24  https://youtube.com/playlist?list=OLAK5uy_k_gjL8bRpuKouEZ1Tjlc8B9l9YsU0VTog -x --audio-format mp3 --audio-quality 0   -o "L5/Mazas36_ClaudioCruz_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 8,10,12,13,20,24  https://youtube.com/playlist?list=OLAK5uy_m-jc7KP-2UTBm0nflYsfTPyu8zAxRk_j4 -x --audio-format mp3 --audio-quality 0   -o "L5/Mazas36_JPRafferty_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 8,10,12,13,20,24  https://youtube.com/playlist?list=PLIcJOrQKnxs1EpGRbrdoM9LTzyDkSYxDQ -x --audio-format mp3 --audio-quality 0   -o "L5/Mazas36_BochanKang_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 8,10,12,13,20,24  https://youtube.com/playlist?list=PLANsNQxncQtggF9zD1XxSV5XeOJt0Oira -x --audio-format mp3 --audio-quality 0   -o "L5/Mazas36_BernardChevalier_%(playlist_index)03d_%(title)s.%(ext)s"




### GRADE 6 ###
yt-dlp --yes-playlist --playlist-items 6-16,19-23,27,29,30 https://youtube.com/playlist?list=OLAK5uy_njnpf7e-zqO6YhbCucfU5SDXOuqFUz8F8 -x --audio-format mp3 --audio-quality 0   -o "L6/Kreutzer_SunKim_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 6-16,19-23,27,29,30 https://youtube.com/playlist?list=PL57E3A778DD644CC5 -x --audio-format mp3 --audio-quality 0   -o "L6/Kreutzer_BernardChevalier_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 5-15,18-22,26,28,29  https://youtube.com/playlist?list=PLIcJOrQKnxs0-_OhkyymPTmzJktkjkNuA -x --audio-format mp3 --audio-quality 0   -o "L6/Kreutzer_BochanKang_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 31,35,40,48,52 https://youtube.com/playlist?list=PLIcJOrQKnxs1EpGRbrdoM9LTzyDkSYxDQ -x --audio-format mp3 --audio-quality 0   -o "L6/Mazas36_BochanKang_%(playlist_index)03d_%(title)s.%(ext)s"
yt-dlp --yes-playlist --playlist-items 31,35,40,48,52,70   https://youtube.com/playlist?list=PLANsNQxncQtggF9zD1XxSV5XeOJt0Oira -x --audio-format mp3 --audio-quality 0   -o "L6/Mazas36_BernardChevalier_%(playlist_index)03d_%(title)s.%(ext)s"

