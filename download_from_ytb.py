from pytube import YouTube
YouTube('https://www.youtube.com/watch?v=SlCw9PErPFQ&ab_channel=HyW').streams.filter(res="720p").first().download('./')