###  SUZUKI  ###
##   book 1   ## all monophonic
yt-dlp  https://www.youtube.com/watch?v=JTzSxS3q9Zs  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Suzuki Violin School Vol. 1"  -o "chapter:monoSuzuki/Suzuki1_BochanKang_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
yt-dlp https://www.youtube.com/watch?v=6Y8yHaKTKuw  -x --audio-format mp3 --audio-quality 0 --split-chapters -o "chapter:monoSuzuki/Suzuki1_YeraLee_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 5-26 https://youtube.com/playlist?list=PLMFLV1svOUsQCTTk8a1tAZG0BYr2CrF4f -x --audio-format mp3 --audio-quality 0   -o "monoSuzuki/Suzuki1_MyriadMSDA_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --postprocessor-args "ffmpeg:-ss 0:0:5" https://youtube.com/playlist?list=PLHyVuM6mAj72llEZGOxYnaC2qtpvhu-yk -x --audio-format mp3 --audio-quality 0   -o "monoSuzuki/Suzuki1_MikeChau_%(playlist_index)03d_%(title)s_[%(id)s]_0005_%(duration)04d.%(ext)s"
rm '[Suzuki Violin School  Book 1] ALL Songs'*
rm 'Suzuki Violin School Book Vol. 1'*
##   book 2   ## mono except 2.9 Gavotte from Mignon
yt-dlp  https://www.youtube.com/watch?v=K6NV1vZhnqU  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Suzuki Violin School Vol. 2" --remove-chapters "Mignon"  -o "chapter:monoSuzuki/Suzuki2_BochanKang_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
yt-dlp https://www.youtube.com/watch?v=M9s6rEct07g  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Mignon" -o "chapter:monoSuzuki/Suzuki2_YeraLee_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-8,10-12 https://youtube.com/playlist?list=PLMFLV1svOUsTEGLJEh7Yt4GSmzColeSY7 -x --audio-format mp3 --audio-quality 0   -o "monoSuzuki/Suzuki2_MyriadMSDA_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-8,10-12 --postprocessor-args "ffmpeg:-ss 0:0:5" https://youtube.com/playlist?list=PLHyVuM6mAj73Fv99uMulYHPd_CrwBPW3z -x --audio-format mp3 --audio-quality 0   -o "monoSuzuki/Suzuki2_MikeChau_%(playlist_index)03d_%(title)s_[%(id)s]_0005_%(duration)04d.%(ext)s"
rm '[Suzuki Violin Book 2] All Songs'*
rm 'Suzuki Violin School Book Vol. 2'*
##   book 3   ## mono except 3.6 and 3.7, though Myriad playlist is messed up with the numbers
yt-dlp  https://www.youtube.com/watch?v=IgnEw3E4_L8  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Suzuki Violin School" --remove-chapters "6.Gavotte"  --remove-chapters "7.Bourrée"  -o "chapter:monoSuzuki/Suzuki3_BochanKang_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=n30OtrMWkrE  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "3-6 Gavotte in DM" --remove-chapters "3-7 Bourree" -o "chapter:monoSuzuki/Suzuki3_YeraLee_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-2,4,6-7 https://youtube.com/playlist?list=PLMFLV1svOUsSEHvPoY64L1machR4j1mrN -x --audio-format mp3 --audio-quality 0   -o "monoSuzuki/Suzuki3_MyriadMSDA_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-5 --postprocessor-args "ffmpeg:-ss 0:0:5" https://www.youtube.com/playlist?list=PLHyVuM6mAj725clT3jMqeem5WDSySjp8g -x --audio-format mp3 --audio-quality 0   -o "monoSuzuki/Suzuki3_MikeChau_%(playlist_index)03d_%(title)s_[%(id)s]_0005_%(duration)04d.%(ext)s"
rm '[Suzuki Violin School Book 3]'*
rm 'Suzuki Violin School Book Vol. 3'*
##   book 4   ## only items 4.4-4.5 (Vivaldi A minor concerto mov1,3) & 4.6 (Bach 2vln concerto mov1) are monophonic
yt-dlp --yes-playlist --playlist-items 4,5,7 --postprocessor-args "ffmpeg:-ss 0:0:5" https://youtube.com/playlist?list=PLHyVuM6mAj71QYqb24xkVwUR_0qTfg228 -x --audio-format mp3 --audio-quality 0   -o "monoSuzuki/Suzuki4_MikeChau_%(playlist_index)03d_%(title)s_[%(id)s]_0005_%(duration)04d.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=Rz5_5hFlSJ8  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Suzuki Violin School" --remove-chapters "F. Seitz"  -o "chapter:monoSuzuki/Suzuki4_BochanKang_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=ttHaG9vM5O8  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Seitz" --remove-chapters "Perpetual Motion" -o "chapter:monoSuzuki/Suzuki4_YeraLee_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
rm 'Suzuki Violin School Book Vol. 4'*
rm '[Suzuki Violin Book 4] All Songs'*
##   book 5   ## only items 5.2, 5.4, 5.5 and 5.7 are monophonic (Country Dance, German Dance, Bach 2vln mov1 vln1)
yt-dlp --yes-playlist --playlist-items 2,6,7,9 --postprocessor-args "ffmpeg:-ss 0:0:5" https://youtube.com/playlist?list=PLHyVuM6mAj72ZUNMbEF3ITNf7r2gAsbYu -x --audio-format mp3 --audio-quality 0   -o "monoSuzuki/Suzuki5_MikeChau_%(playlist_index)03d_%(title)s_[%(id)s]_0005_%(duration)04d.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=TU9J9g783XE  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Suzuki Violin School" --remove-chapters "Gavotte" --remove-chapters "Concerto in G Minor" --remove-chapters "F. M. Veracini" -o "chapter:monoSuzuki/Suzuki5_BochanKang_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=--0L9xoZJb0  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "Gavotte" --remove-chapters "G minor" --remove-chapters "Gigue" -o "chapter:monoSuzuki/Suzuki5_YeraLee_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
rm 'Suzuki Violin Book 5, All Songs'*
rm 'Suzuki Violin School Book Vol. 5 Full Version'*



###  DANCLA  ### the start of the etude there is an exercise. Not suitable for alignment!
# with exerises, total of 25 etudes, but most of the things are ok in these two playlists.
# Remove from Chevalier: 6, 12, 17, 18, 19, 20, 24, 30, 31, 36 (27 recordings)
# Remove from Mantovani: 6, 12, 17, 19, 20, 24, 30, 31, 36 (28 recordings), but further includes 4 bis pieces!
# Mantovani playlist item 39 (studio 14 bis) includes pizzicato, so excluded for now
yt-dlp --yes-playlist --playlist-items 1-5,7-11,13-17,22-25,27-31,34-37 https://youtube.com/playlist?list=PLANsNQxncQtipmZ1Rd_8SwXZ3x88DpoFX -x --audio-format mp3 --audio-quality 0   -o "monoDancla/DanclaOp84_BernardChevalier_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-5,7-11,13-17,19,22-24,26-30,33-36,38,40-41 https://youtube.com/playlist?list=OLAK5uy_m9E9Z9q3bnyLZ8kqmGTwcUd_n3R0Z8j24 -x --audio-format mp3 --audio-quality 0   -o "monoDancla/DanclaOp84_GiovanniMantovani_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"



### WOHLFAHRT ###
# 41 etudes with constant single voice, no pizzicato [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18,
# {19, 20, 22, 23, 24, 25, 26, 29, 31, 32, 34, 35, 36, 37, 39, 41, 42, 45, 48, 49, 50, 51, 52, 57, 58}
# BochanKang plays them with bowing variations, which might be very interesting  research problem on its own right
# BochanKang normal ones: 4,8,9,10,24,29,31,32,35,37,39,42,52
yt-dlp --yes-playlist --playlist-items 4,8,9,10,24,29,31,32,35,37,39,42,52 https://youtube.com/playlist?list=PLIcJOrQKnxs1DNu-hrKQrdvmsbe0kVYlA -x --audio-format mp3 --audio-quality 0   -o "monoWohlfahrt/WohlfahrtOp45_BochanKang_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
# BochanKang bowing variations: 1,2,3,5,6,7,11,12,13,14,15,16,17,18,19,20,21,22,23,25,26,34,36,41
# BochanKang w/ 'lesson' *44,*45,*46,47,48,49(withVariations),50,51,55,56,57,58,59,60 (*keeps talking in actual play)
yt-dlp --yes-playlist --playlist-items 1-12,15-17,18-20,22-26,29,31,32,34-37,39,41-42,48-52,57,58 https://youtube.com/playlist?list=PLIcJOrQKnxs1DNu-hrKQrdvmsbe0kVYlA --split-chapters --remove-chapters "Lesson"  -x --audio-format mp3 --audio-quality 0   -o "chapter:monoWohlfahrt/WohlfahrtOp45_BochanKang_%(playlist_index)03d_%(section_number)03d_%(title)s_%(section_title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
# TimRohwer only till 44, MichaelPijoan only till 38
yt-dlp --yes-playlist --playlist-items 1-12,15-17,18-20,22-26,29,31,32,34-37,39,41-42,45,48-52,57,58 https://www.youtube.com/playlist?list=OLAK5uy_mESjCL-XUZUHoO2T_uXGd-q4YYJhgbLOo -x --audio-format mp3 --audio-quality 0   -o "monoWohlfahrt/WohlfahrtOp45_JPRafferty_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-12,15-17,18-20,22-26,29,31,32,34-37,39,41-42,45,48-52,57,58 https://youtube.com/playlist?list=PLQT2_mTTuV128kC6hL-W2aw5Ntj7Ed3mO -x --audio-format mp3 --audio-quality 0   -o "monoWohlfahrt/WohlfahrtOp45_BernardChevalier_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-12,15-17,18-20,22-26,29,31,32,34-37,39,41-42,45,48-52,57,58 https://youtube.com/playlist?list=PLCA313AAE75E6FEED -x --audio-format mp3 --audio-quality 0   -o "monoWohlfahrt/WohlfahrtOp45_BrianClement_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-12,15-17,18-20,22-26,29,31,32,34-37,39,41-42 https://youtube.com/playlist?list=PLatxbbcShnHWgHACAchZ_EGyZmMAi73mP -x --audio-format mp3 --audio-quality 0   -o "monoWohlfahrt/WohlfahrtOp45_TimRohwer_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-12,15-17,18-20,22-26,29,31,32,34-37 https://youtube.com/playlist?list=PLEF8D2CF5038BBC44 -x --audio-format mp3 --audio-quality 0   -o "monoWohlfahrt/WohlfahrtOp45_MichaelPijoan_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
rm 'Wohlfahrt'*
rm '[Lesson]'*


### SITT ###
# Bernard Chevalier books 1-2, Giovanni Mantovani 1-2-3.
# Book1 first position
yt-dlp --yes-playlist --playlist-items 1-5,8-9,12-13,15,19 https://youtube.com/playlist?list=PLANsNQxncQtiwcogE9RD--GLUBpBU66Sl -x --audio-format mp3 --audio-quality 0   -o "monoSitt/SittOp32Book1_BernardChevalier_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-5,8-9,12-13,15,19 https://youtube.com/playlist?list=OLAK5uy_nSOq7aqdIOZe3zug4sUo2T-1wAigM_ZjA -x --audio-format mp3 --audio-quality 0   -o "monoSitt/SittOp32Book1_GiovanniMantovani_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
# Book2 positions 2-5
yt-dlp --yes-playlist --playlist-items 21,23-24,26,28-29,31-33,35-40 https://youtube.com/playlist?list=PLANsNQxncQtiwcogE9RD--GLUBpBU66Sl -x --audio-format mp3 --audio-quality 0   -o "monoSitt/SittOp32Book2_BernardChevalier_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1,3-4,6,8-9,11-13,15-20 https://youtube.com/playlist?list=OLAK5uy_m6-FNXrMUy8VyaVW1QgH2wj-N3zwkjq0w -x --audio-format mp3 --audio-quality 0   -o "monoSitt/SittOp32Book2_GiovanniMantovani_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
# Book3 positions 6-7
yt-dlp --yes-playlist --playlist-items 3,5,7-10,13,15 https://youtube.com/playlist?list=OLAK5uy_nknVl0sxnVZyJzC2oP3Q-BDbbS_7jZYV0 -x --audio-format mp3 --audio-quality 0   -o "monoSitt/SittOp32Book3_GiovanniMantovani_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"


### KAYSER ### BernardChevalier wrong playlist numbers, MikeChau almost inverted list [::-1] with some wrong numbers
yt-dlp --yes-playlist --playlist-items 1,2,10,12,30 --postprocessor-args "ffmpeg:-ss 0:0:3" https://youtube.com/playlist?list=PLIcJOrQKnxs2L67LkeccL8K9L_Uv8raUo -x --audio-format mp3 --audio-quality 0   -o "monoKayser/KayserOp20_BochanKang_%(playlist_index)03d_%(title)s_[%(id)s]_0003_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1,2,10,12,30 --postprocessor-args "ffmpeg:-ss 0:0:4" https://youtube.com/playlist?list=PLTdeeeVSPtA0-ZdTT--hw4-R8FSG82AOS -x --audio-format mp3 --audio-quality 0   -o "monoKayser/KayserOp20_AlexandrosIakovou_%(playlist_index)03d_%(title)s_[%(id)s]_0004_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1,2,10,12,30 https://youtube.com/playlist?list=PLQyxywuv2Kp40hffFLWpqypWP-XHOaiho -x --audio-format mp3 --audio-quality 0   -o "monoKayser/KayserOp20_FabricioValvasori_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1,2,10,12,30 --postprocessor-args "ffmpeg:-ss 0:0:4" https://youtube.com/playlist?list=PLCtX1MbsedDqblIuqvl33e-sSQBsKRqpn -x --audio-format mp3 --audio-quality 0   -o "monoKayser/KayserOp20_ClaudioCruz_%(playlist_index)03d_%(title)s_[%(id)s]_0004_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1,2,7,10,33 https://youtube.com/playlist?list=PLANsNQxncQtj3ryGK54IjNfp8oZ0OjBwP -x --audio-format mp3 --audio-quality 0   -o "monoKayser/KayserOp20_BernardChevalier_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1,2,10,12,30 https://youtube.com/playlist?list=OLAK5uy_lZgu-rKg2uLygrhs2jqdsGwK2A6qeZbM4 -x --audio-format mp3 --audio-quality 0   -o "monoKayser/KayserOp20_JPRafferty_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 14,25,27,35,36 --postprocessor-args "ffmpeg:-ss 0:0:3" https://youtube.com/playlist?list=PLHyVuM6mAj72oXRcOIAbKDE2D74SXqryg -x --audio-format mp3 --audio-quality 0   -o "monoKayser/KayserOp20_MikeChau_%(playlist_index)03d_%(title)s_[%(id)s]_0003_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1,2,10,12,30 https://youtube.com/playlist?list=PLQT2_mTTuV11hfeuY4ilSyDuNi6bh12je -x --audio-format mp3 --audio-quality 0   -o "monoKayser/KayserOp20_SunKim_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"


### MAZAS ### JPRafferty and ClaudioCruz only have the first 30 etudes
yt-dlp --yes-playlist --playlist-items 8,10,12,13,20,24  https://youtube.com/playlist?list=OLAK5uy_k_gjL8bRpuKouEZ1Tjlc8B9l9YsU0VTog -x --audio-format mp3 --audio-quality 0   -o "monoMazas/MazasOp36_ClaudioCruz_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 8,10,12,13,20,24  https://youtube.com/playlist?list=OLAK5uy_m-jc7KP-2UTBm0nflYsfTPyu8zAxRk_j4 -x --audio-format mp3 --audio-quality 0   -o "monoMazas/MazasOp36_JPRafferty_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 8,10,12,13,20,24,31,35,40,48,52  https://youtube.com/playlist?list=PLIcJOrQKnxs1EpGRbrdoM9LTzyDkSYxDQ -x --audio-format mp3 --audio-quality 0   -o "monoMazas/MazasOp36_BochanKang_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 8,10,12,13,20,24,31,35,40,48,52,70   https://youtube.com/playlist?list=PLANsNQxncQtggF9zD1XxSV5XeOJt0Oira -x --audio-format mp3 --audio-quality 0   -o "monoMazas/MazasOp36_BernardChevalier_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"


### DONT OP 37 ###
# 10 etudes
# {5, 7, 8, 11, 12, 13, 14, 15, 16, 17}
yt-dlp --yes-playlist --playlist-items 5,7,8,11-17 https://youtube.com/playlist?list=PLIcJOrQKnxs3uPaPzTizuYVM0NWd8gsUt -x --audio-format mp3 --audio-quality 0   -o "monoDontOp37/DontOp37_BochanKang_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 5,7,8,11-17 https://youtube.com/playlist?list=PLiFl6GDKYbH5LdMRKyuYAWgh4J7EzIfyr -x --audio-format mp3 --audio-quality 0   -o "monoDontOp37/DontOp37_Eliolin_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 5,7,8,11-17 https://youtube.com/playlist?list=PLYOGF0ZHOdI31As3wz9K8TA4TYU8lWiyF -x --audio-format mp3 --audio-quality 0   -o "monoDontOp37/DontOp37_LeahRoseman_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"


### KREUTZER ### BochanKang doesnt have the etude nr 1
yt-dlp --yes-playlist --playlist-items 1-16,19-23,27,29,30 https://youtube.com/playlist?list=OLAK5uy_njnpf7e-zqO6YhbCucfU5SDXOuqFUz8F8 -x --audio-format mp3 --audio-quality 0   -o "monoKreutzer/Kreutzer_SunKim_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-16,19-23,27,29,30 https://youtube.com/playlist?list=PL57E3A778DD644CC5 -x --audio-format mp3 --audio-quality 0   -o "monoKreutzer/Kreutzer_BernardChevalier_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-15,18-22,26,28,29  https://youtube.com/playlist?list=PLIcJOrQKnxs0-_OhkyymPTmzJktkjkNuA -x --audio-format mp3 --audio-quality 0   -o "monoKreutzer/Kreutzer_BochanKang_%(playlist_index+1)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 1-16,19-23,27,29,30 https://youtube.com/playlist?list=PLQT2_mTTuV13P0LuXKWtNTQjFSZdWT4af -x --audio-format mp3 --audio-quality 0   -o "monoKreutzer/Kreutzer_CihatAskin_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"


### FIORILLO ### 13 etudes, Eliolin has some missing numbers but missing only one monophonic etude
yt-dlp --yes-playlist --playlist-items 3,5,7,8,10,11,14-16,19,21,22,30 https://youtube.com/playlist?list=PLQT2_mTTuV10gRqjEU95VfV1DL09LRFcl -x --audio-format mp3 --audio-quality 0   -o "monoFiorillo/FiorilloOp3_BernardChevalier_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 3,5,7,8,10,11,14-16,19 https://youtube.com/playlist?list=PLHyVuM6mAj72FXhNqZ1XsiFib1UyATvVx -x --audio-format mp3 --audio-quality 0   -o "monoFiorillo/FiorilloOp3_MikeChau_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 3,5,7,8,10,11,13-15,18,19,20 https://youtube.com/playlist?list=PLQT2_mTTuV12qyI1IFY_SRk39wUh2BqG2 -x --audio-format mp3 --audio-quality 0   -o "monoFiorillo/FiorilloOp3_Eliolin_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"


### RODE ### 7 etudes {8, 9, 10, 12, 17, 18, 22}
yt-dlp --yes-playlist --playlist-items 8,9,10,12,17,18,22 https://youtube.com/playlist?list=PLQT2_mTTuV12R3U1VZjnkSyetsTPQjjW9 -x --audio-format mp3 --audio-quality 0   -o "monoRode/RodeOp22_LeahRoseman_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 8,9,10,12,17,18,22 https://youtube.com/playlist?list=PLQT2_mTTuV12JGvHNV4dEr2AHyo2DsuHU -x --audio-format mp3 --audio-quality 0   -o "monoRode/RodeOp22_LiviuPrunaru_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 8,9,10,12,17,18,22 https://youtube.com/playlist?list=OLAK5uy_mfro24wkBcE3kLwCufQ0QOKlj3IPWRWv0 -x --audio-format mp3 --audio-quality 0   -o "monoRode/RodeOp22_AxelStrauss_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 8,9,10,12,17,18,22 https://youtube.com/playlist?list=PLQT2_mTTuV11-ROwQS4UvIwUSL6NCNGSE -x --audio-format mp3 --audio-quality 0   -o "monoRode/RodeOp22_BernardChevalier_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp  https://www.youtube.com/watch?v=4CHbMNJOJkU  -x --audio-format mp3 --audio-quality 0 --split-chapters --remove-chapters "I. C" --remove-chapters "II. A minor"  --remove-chapters "III. G"  --remove-chapters "IV. E"  --remove-chapters "V. D"  --remove-chapters "VI. B"  --remove-chapters "VII. A maj"  --remove-chapters "XI. B "  --remove-chapters "XIII. G"  --remove-chapters "XIV. E"  --remove-chapters "XV. D"  --remove-chapters "XVI. B"  --remove-chapters "XIX. E" --remove-chapters "XX. C"  --remove-chapters " XXI. B" --remove-chapters "XXIII. F" --remove-chapters "XXIV. D" -o "chapter:monoRode/RodeOp22_OscarShumsky_%(section_number)03d_%(section_title)s_[%(id)s]_%(section_start)04d_%(section_end)04d.%(ext)s"
rm 'Pierre Rode by Oscar Shumsky'*


### DONT OP 35 ### 7 etudes {2, 3, 5, 6. 7, 15, 17}
yt-dlp --yes-playlist --playlist-items 2,3,5,6,7,15,17 https://youtube.com/playlist?list=PLANsNQxncQthVJzfzMJ2VpnDKqOepNG4g -x --audio-format mp3 --audio-quality 0   -o "monoDontOp35/DontOp35_BernardChevalier_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 2,3,5,6,7,15,17 https://youtube.com/playlist?list=PLQT2_mTTuV12YO0OL6iNOIm0zyJslRhXW -x --audio-format mp3 --audio-quality 0   -o "monoDontOp35/DontOp35_LeahRoseman_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"


### GAVINIES ### 6 etudes
yt-dlp --yes-playlist --playlist-items 2,6,10-13 https://youtube.com/playlist?list=PLQT2_mTTuV11-56MuXDnhyW7mr3Q7WpkU -x --audio-format mp3 --audio-quality 0   -o "monoGavinies/Gavinies_LeahRoseman_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"
yt-dlp --yes-playlist --playlist-items 2,6 https://youtube.com/playlist?list=PLFshF728MOTZCJG_KJTKY5wxcbkiIZ6HH -x --audio-format mp3 --audio-quality 0   -o "monoGavinies/Gavinies_AliceHallstrom_%(playlist_index)03d_%(title)s_[%(id)s]_0000_%(duration)04d.%(ext)s"


