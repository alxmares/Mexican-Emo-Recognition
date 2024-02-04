import customtkinter as ctk
import pyaudio
import wave
import whisper
from threading import Thread, Event
from pysentimiento import create_analyzer
import transformers
ctk.set_appearance_mode('System')
ctk.set_default_color_theme('blue')


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # Variables
        self.recording=False
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
        
        self.language_label = ctk.CTkLabel(self.right_frame,text='Language = ')
        self.language_label.grid(row=1,column=0,pady=15)
        self.seconds_label = ctk.CTkLabel(self.right_frame,text='Seconds Spoken = ')
        self.seconds_label.grid(row=2,column=0,pady=(0,15))
        self.emotion_label = ctk.CTkLabel(self.right_frame,text='Emotion = ')
        self.emotion_label.grid(row=3,column=0)
    
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
        self.state_label.configure(text='Getting emotion...')
        print('*Getting emotion')
        print(self.text)
        emotion = pysent_model.predict(self.text)
        print(emotion)
        self.state_label.configure(text='')
        self.emotion_label.configure(text='Emotion = ' + str(emotion.output))
        
        
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
        app.emotion_label.configure(text='Emotion = ')
        app.recording = not app.recording
        app.button1.configure(text='Record')
        app.main_label.configure(text='Record')
        app.state_label.configure(text='')
        return

class GetText(Thread):
    def __init__(self,audio_path=r'C:\Users\AlxMa\Documents\Python\output.wav',*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_path=audio_path 
        
    def run(self):
        global whisper_model
        
        print('Converting to text')
        self.result = whisper_model.transcribe(self.audio_path, word_timestamps=True)
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
        return
    
    def show_whisper(self):
        text = self.result['text']
        language = self.result['language']
        seconds = self.result['segments']
        seconds = seconds[0]['end']
        
        text = 'Text = ' + str(text)
        app.textbox.configure(state='normal')
        app.textbox.delete('0.0','end')
        app.textbox.insert('0.0',text)
        app.textbox.configure(state='disabled')
        language = 'Language = ' + str(language)
        app.language_label.configure(text = language)
        seconds = 'Seconds Spoken = '+str(seconds)+'s'
        app.seconds_label.configure(text=seconds)
    
def load_models():
    global whisper_model, pysent_model
    whisper_model = whisper.load_model('base')
    print('*Whisper model loaded')
    
    transformers.logging.set_verbosity(transformers.logging.ERROR)
    pysent_model =  create_analyzer(task='emotion',lang='es')
    print('*Pysentimiento model loaded')
    return               
                
if __name__ == '__main__':
    load_model = Thread(target=load_models)
    app = App()
    load_model.start()
    app.mainloop()