import numpy as np, scipy.io.wavfile as wf
SR=44100; n=int(306*SR); L=np.zeros(n); R=np.zeros(n)
mtof=lambda m:440*2**((m-69)/12)
for i,m in enumerate([74,81,86,90]):
    x=np.arange(6*SR)/SR
    s=(np.sin(2*np.pi*mtof(m)*x)+0.3*np.sin(2*np.pi*mtof(m)*2.76*x)*np.exp(-x*3))*np.exp(-x*0.7)*0.09
    st=int((300+0.05*i)*SR); p=0.3+0.13*i
    m_=min(len(s),n-st); L[st:st+m_]+=s[:m_]*np.cos(p*np.pi/2); R[st:st+m_]+=s[:m_]*np.sin(p*np.pi/2)
wf.write('chime5.wav',SR,(np.stack([L,R],1)*32767).astype(np.int16))
