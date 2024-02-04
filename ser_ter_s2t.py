import customtkinter as ctk
import pyaudio
import wave
import whisper
import keras
from threading import Thread, Event
from pysentimiento import create_analyzer
import transformers

import nltk
from nltk.corpus import stopwords
import string

import librosa 
import numpy as np

ctk.set_appearance_mode('System')
ctk.set_default_color_theme('blue')


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Variables
        self.recording=False
        self.result = []
        self.path=''
        self.text = ''
        
        self.title='Speech to text - Whisper.py'
        self.geometry(f'{1000}x{300}')
        
        self.grid_columnconfigure(1,weight=1)
        self.grid_columnconfigure(2,weight=0)
        self.rowconfigure((0,1,2,3), weight=1)
        
        # Create left and right grid 
        self.left_frame = ctk.CTkFrame(self,width=300,corner_radius=0)
        self.left_frame.grid(row=0,column=0,rowspan=4,sticky='nsew')
        self.left_frame.grid_rowconfigure(5,weight=0)
        self.right_frame = ctk.CTkFrame(self,corner_radius=0, fg_color='transparent')
        self.right_frame.grid(row=0,column=1,rowspan=2,sticky='nswe')
        self.right_frame.grid_rowconfigure(4,weight=0)
        
        # Create left widgets
        self.main_label = ctk.CTkLabel(self.left_frame, text='Record',font=ctk.CTkFont(size=20,weight='bold'))
        self.main_label.grid(row=0,column=0,padx=20,pady=(20,10))
        
        # Record button
        self.button1 = ctk.CTkButton(self.left_frame, command=self.record, text='Record')
        self.button1.grid(row=1,column=0,padx=20,pady=10) 
        
        # Speech to text button
        self.button2 = ctk.CTkButton(self.left_frame,command=self.s2t,text='Convert to text')
        self.button2.grid(row=2,column=0,padx=20,pady=20)
        self.button2.configure(state='disabled')
        
        # Get emotion button
        self.button3 = ctk.CTkButton(self.left_frame,command=self.get_emotion,text='Get Emotion')
        self.button3.grid(row=3,column=0,pady=10)
        self.button3.configure(state='disabled')
        
        # Crate progressbar
        self.slider_frame = ctk.CTkFrame(self.left_frame,fg_color='transparent',width=140)
        self.slider_frame.grid(row=4,column=0,pady=10)
        self.slider_frame.grid_columnconfigure(0,weight=1)
        self.slider_frame.grid_rowconfigure(2,weight=1)
        self.progressbar = ctk.CTkProgressBar(self.slider_frame,width=140)
        self.progressbar.grid(row=1,column=0)
        self.progressbar.set(0)
        
        # State label
        self.state_label = ctk.CTkLabel(self.left_frame,text='',font=ctk.CTkFont(size=10)) 
        self.state_label.grid(row=5,column=0,pady=0)
        
        # right frames
        self.textbox = ctk.CTkTextbox(self.right_frame,width=810,height=150)
        self.textbox.grid(row=0,column=0,padx=5)
        self.configure(state='disabled')
        
        self.words_label = ctk.CTkLabel(self.right_frame,text='Important words = ')
        self.words_label.grid(row=1,column=0,pady=15)
        self.textemo_label = ctk.CTkLabel(self.right_frame,text='Text emotion = ')
        self.textemo_label.grid(row=2,column=0,pady=(0,15))
        self.speechemo_label = ctk.CTkLabel(self.right_frame,text='Speech emotion = ')
        self.speechemo_label.grid(row=3,column=0)
    
    def record(self):
        #print('Record Button')
        if self.recording == False:
            self.recording=not self.recording
            
            # Porgressbar
            self.progressbar.configure(mode='indeterminate')
            self.progressbar.start()
            
            self.button1.configure(text='Stop Recording')
            self.main_label.configure(text='Recording...')
            self.state_label.configure(text='Recording...')
            
            # Threading
            self.event = Event()
            self.record_thread = Recorder(self.event,seconds=10)
            self.record_thread.start()
            return
        # Stop Recording
        else:
            self.event.set()
            return
        
    def s2t(self):
        self.s2t_thread = GetText()
        self.s2t_thread.start()
        
        self.main_label.configure(text='Converting...')
        self.progressbar.configure(mode='indeterminate')
        self.state_label.configure(text='Converting...')
        self.progressbar.start()
        return 
    
    def get_emotion(self):
        self.main_label.configure(text='Getting...')
        self.progressbar.configure(mode='indeterminate')
        self.state_label.configure(text='Getting emotions...')
        self.progressbar.start()
        
        self.ge_thread = GetEmotion(result= self.result)
        self.ge_thread.start()
        
        #print('*Getting emotion')
        #print(self.text)
        #emotion = pysent_model.predict(self.text)
        #print(emotion)
        #self.state_label.configure(text='')
        ###self.emotion_label.configure(text='Emotion = ' + str(emotion.output))
        
        
class Recorder(Thread):
    def __init__(self, event, seconds,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.event = event
        self.seconds = seconds
        self.filename = 'output.wav'
        self.chunk=1024
        self.sample_format=pyaudio.paInt16
        self.channels=2
        self.fs=44100
    
    def run(self):
        print('Recording') 
        while True:
            self.p=pyaudio.PyAudio()
            stream = self.p.open(format=self.sample_format,
                        channels=self.channels,
                        rate=self.fs,
                        frames_per_buffer=self.chunk,
                        input=True)
            self.frames=[]
            
            for i in range(0, int(self.fs / self.chunk * self.seconds)):
                data = stream.read(self.chunk)
                self.frames.append(data)
                
                # Si se presiona el botón de terminar
                if self.event.is_set():
                    print('Thread finished')
                    stream.stop_stream()
                    stream.close()
                    self.p.terminate()
                    print('Record_finished')
                    
                    self.save_audio()
                    return
                
            # Stop and close the stream 
            stream.stop_stream()
            stream.close()
            print('Record_finished')
            
            # Terminate the PortAudio interface
            self.p.terminate()
            
            self.save_audio()
            return

    def save_audio(self, ):
        # Save the recorded data as a WAV file
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        print('File saved')
        
        app.progressbar.stop()
        app.progressbar.configure(mode='determinate')
        app.progressbar.set(0)
        app.button2.configure(state='normal')
        app.button3.configure(state='disabled')
        
        app.words_label.configure(text='Important words =')
        app.textemo_label.configure(text='Text emotion =')
        app.speechemo_label.configure(text='Speech emotion =')
        
        app.recording = not app.recording
        app.button1.configure(text='Record')
        app.main_label.configure(text='Record')
        app.state_label.configure(text='')
        return

class GetText(Thread):
    def __init__(self,audio_path='output.wav',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_path=audio_path 
        
    def run(self):
        global whisper_model
        
        print('Converting to text')
        self.result = whisper_model.transcribe(self.audio_path, word_timestamps=True, fp16=False, language='Spanish')
        self.show_whisper()
        print('Finished')
        print(self.result)
        # Frames
        app.progressbar.stop()
        app.progressbar.configure(mode='determinate')
        app.progressbar.set(0)
        app.main_label.configure(text='Record')
        app.state_label.configure(text='')
        app.button3.configure(state='normal')
        app.text = self.result['text']
        app.result = self.result
        return
    
    def show_whisper(self):
        text = self.result['text']
        
        text = 'Text = ' + str(text)
        app.textbox.configure(state='normal')
        app.textbox.delete('0.0','end')
        app.textbox.insert('0.0',text)
        app.textbox.configure(state='disabled')
        
class GetEmotion(Thread):
    def __init__(self,audio_path='output.wav',sr=16000,target_length=20000,result=[],*args, **kwargs):
        super().__init__(*args, **kwargs)

        self.result = result
        self.text = self.result['text']
        self.audio_path = audio_path
        self.sr = sr
        self.target_length = target_length
        
    def run(self):
        important_segments = {'audio_path': '', 'word': [], 'segment': []}
        important_segments['audio_path'] = self.audio_path
        
        self.important_words = self.remove_stopwords()
        #print(self.important_words)
        self.words = []
        for segment in self.result['segments']:
            for word in segment['words']:
                self.words.append(word)
        #print(self.words)
        
        audio_segments = self.get_important_words()
        
        for word,segment in zip(self.important_words, audio_segments):
            important_segments['segment'].append(segment)
            important_segments['word'].append(word)
        
        #print(important_segments)
        
        # Predicción de emoción en texto en el habla
        self.predictions = self.get_emotions(important_segments)
        self.text_emotion = pysent_model.predict(self.text)
        print(self.predictions)
        
        self.display()
        
        app.progressbar.stop()
        app.progressbar.configure(mode='determinate')
        app.progressbar.set(0)
        app.main_label.configure(text='Record')
        app.state_label.configure(text='')
        return
    
    def display(self):
        # Getting important words its emotions
        important_words_text = 'Important words = '
        speech_emotions = 'Speech emotion = '
        for word,prediction in zip(self.predictions['word'], self.predictions['prediction']):
            important_words_text = important_words_text + str(word) + ', '
            speech_emotions = speech_emotions + str(prediction) + ', '
        print(important_words_text)
        print(speech_emotions)
        
        # Text emotion
        app.words_label.configure(text = important_words_text[:-2])
        
        # Text emotion
        text_emotion = str(self.text_emotion.output)
        if text_emotion == 'others':
            text_emotion = 'neutral'
        app.textemo_label.configure(text = 'Emotion in text = ' + text_emotion)
    
        # Speech emotion
        app.speechemo_label.configure(text = speech_emotions[:-2])
            
        
    def remove_stopwords(self):
        stop_words = set(stopwords.words('spanish'))
        words = nltk.word_tokenize(self.text)
        words = [word for word in words if word.lower() not in stop_words]
        words = [word for word in words if word not in string.punctuation]
        return words
    
    def get_important_words(self):
        audio_segments=[]
        self.audio,sr = librosa.load(self.audio_path,sr=self.sr)
        print('palabras a buscar: ', self.important_words)
        for word in self.words:
            word_to_found = word['word'][1:]  # La palabra se expresa como ' prueba' en lugar de 'prueba'
            word_to_found = ''.join(letter for letter in word_to_found if letter.isalnum()) # Quitar caracteres especiales
            if word_to_found in self.important_words:
                print('Palabra encontrada: ',word_to_found)
                segment = [int(float(word['start'])*sr), int(float(word['end'])*sr)]
                # Mostar las palabras importantes
                print('Segmento: ', segment)
                if segment[0] > 200:
                    segment[0] -=200
                if segment[1] < len(self.audio)-1000:
                    segment[1] +=1000
                else:
                    segment[1] +=1000
                print('Segmento aumentado: ', segment)
                audio_word = self.audio[segment[0]:segment[1]]
                audio_segments.append(audio_word)
        return audio_segments
    
    def fix_length(self,audio):
        audio_len = len(audio)
        if audio_len > self.target_length:
            audio = (audio[:self.target_length])
        else:
            padding_length = self.target_length - audio_len
            audio_padded = np.pad(audio,(0,padding_length))
            audio = (audio_padded)
        return audio
            
    def feature_space(self,audio,n_mels=128,n_fft=1569,hop_length=512):
        mel_spec = librosa.feature.melspectrogram(y=audio,sr=self.sr,n_mels=n_mels,
                                                  n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec,ref=np.max)
        return mel_spec_db
    
    def get_emotions(self,segments):
        global mesd_model
        label_map = {0:'anger', 1:'disgust', 2:'fear', 
                 3:'happiness', 4:'neutral', 5:'sadness'}
        predictions = {'word':[], 'prediction':[], 'pred_array': []}
    
        for i,important_word in enumerate(segments['word']):
            fixed_audio = self.fix_length(segments['segment'][i])
            mel_spec = self.feature_space(fixed_audio)
            mel_spec = np.expand_dims(mel_spec, axis=0)
            prediction = mesd_model.predict(mel_spec)
            
            # decoder
            pred_int = [(max(enumerate(prediction[0]), key=lambda x: x[1])[0])]
            pred_label = [label_map[l] for l in pred_int]
            predictions['word'].append(important_word)
            predictions['prediction'].append(pred_label[0])
            predictions['pred_array'].append(prediction[0])
        return predictions
    
def load_models():
    global whisper_model, pysent_model, mesd_model
    whisper_model = whisper.load_model('base')
    print('*Whisper model loaded*')
    
    transformers.logging.set_verbosity(transformers.logging.ERROR)
    pysent_model =  create_analyzer(task='emotion',lang='es')
    print('*Pysentimiento model loaded*')
    
    mesd_model = keras.models.load_model('mexican_emo_ser/models/model2_cnn2d.h5')
    librosa.load('output.wav')
    print('*Model (2D-CNN) loaded*')
    return               
                
if __name__ == '__main__':
    load_model = Thread(target=load_models)
    app = App()
    load_model.start()
    app.mainloop()