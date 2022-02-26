# violinPedagogue
supervising MIR through pedagogical violin repertoire and curricula

download_dataset.sh script creates the dataset with 6 levels according to the "Violin Methods and Etudes" curriculum described in https://www.violinmasterclass.com/p/violin-methods-and-etudes
The current state of dataset is especially tailored for monophonic pitch extraction. Thus any piece that include double stops or chords are removed.

the pipeline starts with extracting the pitch for the beginner violin pieces (L1) using crepe pitch tracker. 

synth.py is for generating the perfectly annotated tracks based on pitch estimate & refinement.

then in a loop alongside the curriculum L1->L2->L3->L4->L5->L6:
  utils/synth2tfrecord.py is for creating the tfrecord dataset out of synthetic data
  retrain crepe for one grade
  extract with the retrained pitch tracker
  
The resulting pitch extractor is highly tailored for the target instrument
