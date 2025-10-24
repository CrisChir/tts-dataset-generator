# --- IMPORTS TO ADD AT THE TOP OF THE SCRIPT ---
from pyannote.audio import Pipeline
from moviepy.editor import AudioFileClip
# (You need the original audio path and segment start times)

# --- NEW GLOBAL VARIABLE (requires HuggingFace login) ---
PYANNOTE_PIPELINE = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=True # Requires token from huggingface-cli login
)
if torch.cuda.is_available():
    PYANNOTE_PIPELINE.to(torch.device("cuda"))

# ----------------------------------------------------------------------
def transcribe_audio_files(audio_dir,
                           output_csv_path="metadata.csv",
                           ljspeech=False,
                           model_name="large",
                           language_="en",
                           # ⚠️ ADD A NEW ARGUMENT FOR THE FULL AUDIO PATH
                           full_audio_path=None): 
    
    # ... (Lines 34-64: Finding WAV files and loading Whisper model remain the same) ...

    # --- NEW: 1. RUN DIARIZATION ON THE FULL AUDIO FILE ---
    if full_audio_path:
        logger.info(f"Running diarization on full audio: {full_audio_path}")
        try:
            diarization = PYANNOTE_PIPELINE(full_audio_path)
            
            # Extract segment start times (in seconds) from the original filenames
            # Assuming filenames are in a predictable format like '000000_123.45.wav' 
            # where 123.45 is the start time.
            segment_times = {}
            for filename in wav_files:
                # ⚠️ YOU MUST ADAPT THIS LOGIC TO GET THE START TIME 
                # FROM HOW YOUR SEGMENTS WERE CREATED
                try:
                    # Example: Assuming filename '000001_12.34.wav' means segment starts at 12.34s
                    start_time = float(filename.split('_')[-1][:-4]) 
                    segment_times[filename] = start_time
                except:
                    segment_times[filename] = 0.0 # Default to 0 if parsing fails
            
            logger.info("Diarization complete. Mapping speakers to segments.")
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            diarization = None # Continue without diarization

    # --- Transcription Process ---
    metadata_text = []
    metadata_audio_path = []
    # ⚠️ ADD A LIST FOR SPEAKER METADATA
    metadata_speaker = []
    total_files = len(wav_files)
    start_time_total = time.time()

    # Process each audio file
    for i, filename in enumerate(wav_files):
        file_path = os.path.join(audio_dir, filename)
        logger.info(f"Processing file {i+1}/{total_files}: {filename}...")
        start_time_file = time.time()
        text = "" 
        current_speaker = "Unknown" # Default speaker label
        
        # --- NEW: 2. ASSIGN SPEAKER ID ---
        if diarization and filename in segment_times:
            start_sec = segment_times[filename]
            
            # Find the speaker ID from the diarization result that matches the segment's start time
            for turn, speaker in diarization.itertracks(yield_label=True):
                # Check if the segment starts within the speaker's turn
                if turn.start <= start_sec < turn.end:
                    current_speaker = speaker
                    break
        
        # --- 3. WHISPER TRANSCRIPTION ---
        try:
            result = model.transcribe(file_path, language=language_)
            text = result["text"].strip()
            end_time_file = time.time()
            print(f"Done ({end_time_file - start_time_file:.2f}s). Speaker: {current_speaker}. Transcription: '{text}'")

        except Exception as e:
            # ... (Error handling remains the same) ...

        # Store the result (filename without path, transcription, speaker)
        metadata_text.append(text)
        metadata_audio_path.append(filename[:-4])
        metadata_speaker.append(current_speaker) # Store the speaker label

    end_time_total = time.time()
    logger.info(f"Finished processing all files in {end_time_total - start_time_total:.2f} seconds.")

    # --- 4. MODIFY CSV OUTPUT TO INCLUDE SPEAKER ID ---
    # The ljspeech format does not typically include speaker IDs, so we focus on the else block
    if ljspeech:
        # ... (ljspeech block remains the same, without speaker ID) ...
        pass
    else:
        logger.info(f"Writing transcriptions (with speaker IDs) to: {output_csv_path}")
        try:
            with open(output_csv_path, 'w', encoding='utf-8') as f:
                for i in range(len(metadata_audio_path)):
                    # ⚠️ MODIFIED LINE TO INCLUDE SPEAKER ID
                    f.writelines(f'wavs/{metadata_audio_path[i]}.wav|{metadata_speaker[i]}|{metadata_text[i]}\n') 

            logger.info("Metadata CSV file created successfully.")
            return True
        except Exception as e:
            # ... (Error handling remains the same) ...
            return False
