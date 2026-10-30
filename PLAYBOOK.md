# ZonelyTime TikTok Playbook

Oct 23, 2026 · @Lewis

## How this doc works

Every rule carries one of four tags, strongest first:

- **TikTok** — TikTok's own published wording
- **Study** — a large third-party dataset (millions of posts), cited in Sources
- **Ours** — our own analytics
- **Hunch** — a reasonable bet, not yet proven

A Hunch is upgraded only after an Open test shows a clear difference across at least three posts. Popular "3-second rule" figures ("70% retention = 2.2x views" and similar) are left out: the pages quoting them name no source.

What TikTok confirms about ranking: finishing a longer video is a strong signal, captions, sounds and hashtags count as video information, and follower count is not a direct factor. It also avoids showing two videos from the same creator back to back. TikTok's documented signals: [TikTok Newsroom, 2020](https://newsroom.tiktok.com/how-tiktok-recommends-videos-foryou-eu?lang=en-GB).

## Project handover

Start each new chat with: "Read the ZonelyTime TikTok Playbook doc and the README in the C4comp repo folder for the task, then carry on with \<task>." Working files from an old chat don't carry over, so anything still needed lives in the GitHub repo [lewisjakewhite26/C4comp](https://github.com/lewisjakewhite26/C4comp).

Never clone the whole repo: it's about 5 GB of videos. Clone only the latest version of the playbook and the one folder the task needs (a few seconds, under 100 MB):

```
git clone --depth 1 --filter=blob:none --sparse https://github.com/lewisjakewhite26/C4comp.git
cd C4comp && git sparse-checkout set <folder>
```

**Working rules**

- UK English. Answer questions without building until Lewis says go.
- Ask before installing anything. Look at every graphic before delivering it.
- Files on GitHub stay under 100 MB. Push with git; Releases can't be made from Claude.
- No Pixabay audio files in the public repo. No Suno music in monetised videos. Lyria mixes are cleared.
- Tick the AI disclosure on every upload. Check what's already posted before making something new.
- Canva images can't be pulled into Claude's workspace: Lewis has to download and attach them.
  - After any change to this doc, export it to PLAYBOOK.md in the repo and push.

**Engines:** every render, physics and audio script is saved in `engines/` in the repo (see `engines/README.md`): firefly jar, firefly tower, snowflake crystal, ring escape and timers, capybara loops, the 2-hour music mixer, rain bed, jellyfish and lava timers, and the older scene loops. Start new videos from these rather than from scratch, add `engines/` to the sparse checkout when needed, and save any new engine there. Pymunk (2D physics) is installed and approved for escape, bounce and pile-up Shorts. Taichi (sand, snow, water) and Blender are options for later, so ask before installing.

**Audio cleared:** Lyria, Meta AI and ElevenLabs (paid plan, including sound effects). Never Suno or Pixabay in monetised videos.

**In progress**

| Project | State | Waiting on | Repo folder |
| --- | --- | --- | --- |
| Capybara blanket scene (Short, then a 2-hour video) | HIGH PRIORITY. Lewis's three reference images saved in refs/ideas/; next build | Lewis's go-ahead to start | not started |
| Classroom timers (spirograph etc.) | Parked: could work, not for now | Lewis picking it back up | not started |
| Reference ideas Lewis liked: orange cloud timelapse (@adiebee19), cherry-blossom canal walk, ring-escape ball game and "200 balls, one hair apart / butterfly effect" physics Short (@the.viral.potato); clips and notes in refs/ideas/ | Ideas only | Lewis choosing which to build | not started |
| Dropped: long firefly video (lighting too painful) and long maths-morph video (wouldn't be good) | Dropped | Nothing | n/a |
| Window wall Short ("someone's cat is up late") | Video approved (v3) | A better rain recording from Lewis, then mix and export | window-wall-short |
| Nordic meadow long video | Silent 15-min video and 30-min music done | A waterfall recording, then join them | nordic-final, nordic-audio |
| Port in the Storm, 30–60 min | Planned | 3–5 more Gemini harbour clips and a storm recording | not started |
| Aurora river long version | Idea | More flythrough clips | not started |
| Rain video | Idea | A real rain recording | rain-house-loop |
| Cosy cat-on-cushion Short (Gemini clip) | Candidate | Lewis's go-ahead; needs a tall reframe and a hook | not started |
| Storm bedroom (Meta) | Needs regenerating | A new Meta clip without the fade-out and indoor rain | not started |
| Halloween lodge long video, published on YouTube as "Cosy Autumn Cabin with Capybara" (3 h 13 min: 161 copies of the 72 s loop, 58-track music-only mix, levelled to -14 LUFS) | Live. Content ID claim from WMG ("Vspominay" by Jah Khalib) on 0:00–0:15, the drums-only opening of The Corner Room (Lyria). Analysis: same tempo (~84 BPM) and similar kick, but no shared recording, melody or chords. Dispute sent 27 Oct 2026 ("misidentified", original AI-generated music) | WMG's reply (they have 30 days, until about 26 Nov). If released: done. If reinstated: trim 0:00–0:15 in YouTube Studio rather than appeal (an appeal risks a strike), and Claude makes the tracklist with every time 15 s earlier | not in repo (mix is 280 MB) |
| Capybara infinite zoom loop (pond → dewdrop → mug → coffee drop → pond) | PAUSED. Flow keyframe plan and prompts written | Lewis making the K2–K4 keyframes in Flow | refs/ideas/capy_zoom_loop.md |
| Music visualiser videos: Lewis liked the visualiser on Lyria's "The Corner Room" video (a dotted particle diamond on black that pulses and morphs with the music). Idea: simple music-led visualiser videos for the lofi tracks | Idea | Lewis choosing when to build | not started |
| Raindrop race (single window): 4 drops on a rainy night window over a blurred city, each with a soft colour glow (teal, pink, gold, lavender). Slow, relaxing, realistic stop-start sliding with wet trails; drops swallow resting droplets and speed up. First to the line glows and grows. "pick your raindrop 👇" hook. TikTok single race (20–40 s); YouTube long version with race after race and a win tally per colour. Coded, not generated | Planned | Lewis's go-ahead; Claude builds a one-race physics test first (does it move like real rain?) | not started |
| Raindrop race tournament (split screen): left = New York high-rise apartment window (drops fall straight, slow); right = motorway passenger window at night (wind pushes drops sideways, faster, jittery). Semi-final on each window (4 drops each), then a face-off final between the two winners with one finish line across both, then a champion moment and scoreboard. About 2.5–3 min on YouTube; TikTok as the final alone or part 1 / part 2 | Planned | Two Gemini background images from Lewis (in focus, no rain on the glass); decide colour names vs. little drop names; built after the single-window version | not started |

## The spec

The first second decides everything: on the Japanese house post, most viewers left at 0:01 and the average watch was 2.9s of 18s (Ours).

| Feature | Rule | Evidence |
| --- | --- | --- |
| Length (main Shorts) | 7–10s, seamless loop | Study: in 8M TikToks, videos under 30s had the highest engagement rate (6.0% of views). Study: winning short videos had a median length of 9s, with 56% under 10s (commerce-leaning sample). |
| Length (long versions) | Also post some 60s+ "stay a while" cuts | Study: average views rise with length, with 0–30s at 1,200 and 120–180s at 12,000 (correlation, not cause). TikTok: only videos over 1 minute earn Creator Rewards. |
| First frame | Visible motion at 0:00: lightning, a swinging chime, a drip, a twitching ear | Ours: most viewers left at 0:01. Fix: Hunch. |
| On-screen text | One line from frame one, a feeling not a description, e.g. "pov: it's raining and you've got nowhere to be" | Hunch: no large study isolates text overlays |
| Sound | A hook at 0:00 (chime, thunder roll), no fade-in; music the main layer, ambience under it | Hunch. TikTok: sounds count as video information. |
| Loop | Last frame blends into the first, no visible jump | Hunch. TikTok: finishing a video is a strong signal, and short loops are easier to finish. |
| Originality | Every post adds clear creative work (loop, grade, sound design, text); no visible watermark or logo from any AI tool | TikTok: unoriginal, minimally edited or watermarked reused clips are ineligible for the For You feed |
| AI label | Always switch on TikTok's AI-generated label | TikTok: required for AI-generated realistic-looking scenes; unlabelled posts can be removed or restricted |
| Format | 1080×1920, 24fps, H.264, −14 LUFS | Platform standard |
| Music | Cleared tracks only (Lyria mixes); no Suno, no Pixabay in the public repo | House rule |

## What content wins

On our account, cosy weather scenes beat everything else: every post above 600 views is rain, storm, snow or night. Timers trail them at 250–450 (Ours). No large study covers the ambient lo-fi niche specifically, so this section rests on our own numbers.

1. **Rain, storms and night with a warm light** (Japanese house 727, lighthouse storm 701, blizzard hut 656, rainy night lamp 615)
2. **Japanese and anime-style streets** (snowy Japanese suburb 612; the only search that found us was "lofi japan")
3. **Calm landscapes at dusk** (valley falls asleep 681; one post only)
4. **Timers** (jellyfish 447, ink 254): lower reach on TikTok, but over a minute long, so eligible for Creator Rewards later
5. **The cat as a regular character**: Hunch, since cat and no-cat posts score the same so far

Audience (Japanese house post): 80% women, 25–34 the largest age group, 60% UK. The Japanese house post got all its views on day one, in line with a large study finding 96% of a post's views arrive within its first 10 days.

**New lead format (Ours, one post so far):** the \~60-second "satisfying build" Short. The firefly jar got 7x the views of any earlier YouTube Short despite a lower stayed-to-watch rate (16.9% vs 19.9%), which suggests total watch time from a long, building payoff matters more than the first-second swipe rate. Confirm with the retention graph and two or three more posts in the series.

## Captions, hashtags, timing and frequency

Post every day at about 6:30pm UK time. Our viewers peak between 7 and 8pm (Ours), and a 2.3M-post study puts the global peak at 6–9pm, with 8pm the top hour (Study).

| Lever | Rule | Evidence |
| --- | --- | --- |
| Frequency | 1 post a day, at least 6 a week | Study (11M+ TikToks): compared with posting once a week, views per post were 17% higher at 2–5 posts a week, 29% at 6–10 and 34% at 11+ |
| Time | About 6:30pm UK time; recheck the Viewers tab monthly | Ours + Study. Another 7.1M-post study ranks Sunday 9am best and says to trust your own analytics first. |
| Caption | Short, lower case, a mood not a description, with a search word: lofi, japan, rain, cosy | TikTok: captions count as video information. Ours: "lofi japan" was our only search query. |
| Hashtags | 3–5, niche over broad, e.g. #lofi #rain #cosy #japan #rainsounds | Study: posts with hashtags got 5% more views and about 10% more interactions. Hunch: no good data on the ideal number. |
| Call to action | Ask a gentle question ("where would you rather be right now?"). Never ask for likes. | Study: questions brought 26% more comments; asking for likes cut interactions by 60% |
| Follow prompt | One line, e.g. "full hour on YouTube" | Hunch. Followers are 0.2% of viewers so far (Ours). |
| Comments | Reply to every one, ideally within the first hour | Hunch |
| Pins | Pin the three best posts | Hunch |

## Getting paid on TikTok

Creator Rewards is the main TikTok payout, and it is a long way off: it needs 10,000 followers, and we have 15.

| Requirement | Detail |
| --- | --- |
| Followers | 10,000+ |
| Views | 100,000+ in the last 30 days |
| Age and account | 18+, personal (not Business) account in good standing, UK eligible |
| Videos that earn | Over 1 minute long and original only |
| What raises pay | Search value, play duration and finish rate, engagement, originality |
| Typical rate | About $0.40–$1 per 1,000 qualified views (third-party estimate; TikTok publishes no rate) |

Two consequences for us. The 7–10s loops grow the account but will never earn, so the 60s+ versions and the timers are the content that will eventually pay. In the meantime, YouTube long-form remains the better earner.

## Open tests

Each test runs over at least three posts of each kind. Judge it 7 days after posting on average watch time, full-view %, and likes per 100 views.

| Test | Compare | Win looks like | Result |
| --- | --- | --- | --- |
| Length | 7–10s loop vs 60s+ version of a similar scene | Higher views per post; loop should win on full-view % |  |
| Motion at 0:00 | Action in the first frame vs a slow start | Average watch above 5s |  |
| Text overlay | A pov line vs no text | Longer average watch, more likes per view |  |
| Question in caption | A question vs no question | More comments per view |  |
| The cat | Cat vs no cat in a similar scene | Clearly more likes per view |  |
| Follow prompt | Line in the caption vs none | More followers per 1,000 views |  |

- [ ] Send the stats for "Alone with the cat" to start the cat test

## Video log

Add a row for every post. Fill in the stats 7 days after posting, by which point nearly all of a post's views have arrived. Blank cells are still to collect.

| Video | Posted | Length (s) | Views | Avg watch (s) | Full views (%) | Likes | Cat? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Firefly jar, "every firefly that escapes adds two more" (YouTube Short) | 25 Oct | 62 | 1,161 in 3 h (best ever on YouTube; 2 push waves; 97% Shorts feed; 25% of watch time on TV; stayed to watch 16.9%) | pending |  |  | No |
| Port in the storm (YouTube Short) | \~22 Oct | 10 | 159 (1 push wave, then flat; stayed to watch 19.9%) | 27 (250%) |  |  | No |
| Japanese house · rain & lo-fi | 18 Oct | 18 | 727 | 2.9 | 1.35 |  | Yes |
| Watch the valley fall asleep | 20 Oct |  | 681 |  |  |  | No |
| A minute in the deep sea (jellyfish timer) | 20 Oct | 60 | 447 |  |  |  | No |
| Lighthouse in a storm (pixel art) | 17 Oct |  | 701 |  |  |  | No |
| Three minute timer: ink springs | 17 Oct | 180 | 254 |  |  |  | No |
| Blizzard hut |  |  | 656 |  |  |  | No |
| Alone with the cat (tower block) |  |  |  |  |  |  | Yes |
| Rainy night, lamp on, nowhere to be | \~9 Oct |  | 615 |  |  |  | No |
| Snowy night in a quiet Japanese suburb | \~9 Oct |  | 612 |  |  |  | No |

## Pre-post checklist

- [ ] Not a repeat of something already posted
- [ ] 7–10s seamless loop (or a deliberate 60s+ version)
- [ ] Something visibly moving at 0:00
- [ ] One line of on-screen text from frame one
- [ ] Sound hook at 0:00; music louder than the ambience
- [ ] Cleared music only
- [ ] No AI-tool watermark or logo anywhere in the frame
- [ ] Clear creative work added (loop, grade, sound, text), not a raw generated clip
- [ ] Caption has a search word and a gentle question; 3–5 hashtags; no request for likes
- [ ] AI-generated label switched on
- [ ] Posting at about 6:30pm UK time
- [ ] Row added to the video log

## Sources

All opened 23 October 2026. Third-party studies come from social-media tool companies, and their methods are only partly published.

- [How TikTok recommends videos #ForYou](https://newsroom.tiktok.com/how-tiktok-recommends-videos-foryou-eu?lang=en-GB) — TikTok Newsroom, June 2020
- [TikTok Community Guidelines (archived copy)](https://github.com/OpenTermsArchive/contrib-versions/blob/main/TikTok/Community%20Guidelines.md) — AI labelling and For You feed eligibility, via Open Terms Archive
- [New labels for disclosing AI-generated content](https://newsroom.tiktok.com/new-labels-for-disclosing-ai-generated-content?lang=en) — TikTok Newsroom, September 2023
- [TikTok's new guidelines (August 2025 changes)](https://techcrunch.com/2025/08/15/tiktoks-new-guidelines-add-subtle-changes-for-live-creators-ai-content-and-more/) — TechCrunch
- [Social media video statistics 2026](https://www.socialinsider.io/ar/social-media-benchmarks/social-media-video-statistics) — Socialinsider: 8M TikToks, January–June 2026, brand accounts
- [TikTok Study 2026](https://metricool.com/tiktok-trends/) — Metricool: 2.3M posts, 92,000 accounts
- [Best time to post on TikTok](https://buffer.com/resources/best-time-to-post-on-tiktok) — Buffer: 7.1M posts, September 2026
- [TikTok algorithm guide](https://buffer.com/library/tiktok-algorithm/) — Buffer, December 2025, including posting-frequency data from 11M+ TikToks
- [Winning TikTok hook study](https://riffkit.ai/research/tiktok-hook-study) — Riffkit: 1,111 curated videos, August 2026; vendor research, commerce-leaning
- [Creator Rewards Program launch](https://www.socialmediatoday.com/news/tiktok-announces-revamped-creator-rewards-program-to-incentivize-longer-u/710639/) — Social Media Today, March 2024
- [How to join Creator Rewards in 2026](https://www.toptal.com/creator/post/how-to-join-the-tiktok-creator-rewards-program) — Toptal, current eligibility and rate estimates
