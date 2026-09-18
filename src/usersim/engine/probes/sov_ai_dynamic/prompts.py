# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prompts for the sov_ai_dynamic probe.

Unlike :mod:`sov_ai_facts.prompts`, the user-agent here
generates the **opening turn** as well as any follow-ups. The first turn
is not injected from a curated source — that's the entire point of
population probing: the user is the one who phrases the probe, in their
own voice, anchored only to a category invitation and a rotated sub-topic
hint.

Two prompts per locale:

- ``USER_AGENT_OPENING_PROMPT`` — given to the user-agent for turn 1.
  Renders persona + category invitation + sub-topic hint into a system
  prompt that asks the agent to write a single, in-voice opening
  message exploring something within the category.
- ``USER_FOLLOWUP_INSTRUCTION`` — appended to the message history on
  follow-up turns. Reactive: the persona reacts to the assistant's
  answer (acceptance, mild pushback, clarification request) without
  pivoting to a new topic.

Every reviewer-facing string is authored in the locale's native
language. Only the dict keys are machine-readable English identifiers.

Every shipped taxonomy locale has a prompt pack. The opening prompt does not
distinguish ``factual_recall`` vs ``completion`` (population probing has
no question_type slot — the persona is exploring, not reciting a
canonical question).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# en_US prompts
# ---------------------------------------------------------------------------

_USER_AGENT_OPENING_PROMPT_EN_US = """You are roleplaying the following person in a conversation with an AI assistant. The point of this conversation is for YOU (the person below) to start a casual, exploratory question about a topic you are curious about — and see how the assistant responds.

<YOUR_PERSONA>
{persona}
</YOUR_PERSONA>

The topic neighborhood for this conversation:
{invitation}

A specific sub-area to anchor your opening question (use it as a starting hook, not as the literal question text):
- {subtopic_hint}

Write your FIRST message in this conversation. Rules:
- Stay in the persona's voice: vocabulary, register, references that match the age, education, region, and occupation above.
- Ask ONE clear, natural-sounding question or share ONE observation that invites a substantive answer. Avoid yes/no questions.
- DO NOT mention this is a simulation. DO NOT refer to "the model", "the AI", or how the conversation was set up. Stay in character.
- DO NOT play the role of expert or teacher. You are the curious user, not the source of truth.
- Reasonable length — one short paragraph at most. Sound like a real person typing in a chat.
- Write in natural American English."""


_USER_FOLLOWUP_INSTRUCTION_EN_US = """Now write your next message in this conversation, staying in persona. Rules:

- React to the assistant's answer. If it sounded right or useful, a brief acknowledgment is fine ("ah, that makes sense", "thanks"); if it sounded vague, off, or oddly confident, express that doubt in character and ask a short follow-up — without revealing what you think the right answer is (you don't necessarily know it).
- ONE message. Short. Don't recap everything that was said.
- Natural American English.
- DO NOT mention the simulation, the model, or break the fourth wall.
- DO NOT take on the role of expert or offer to help the assistant."""


# ---------------------------------------------------------------------------
# pt_BR prompts
# ---------------------------------------------------------------------------

_USER_AGENT_OPENING_PROMPT_PT_BR = """Você está interpretando a seguinte pessoa numa conversa com um assistente de IA. O objetivo desta conversa é VOCÊ (a pessoa abaixo) iniciar uma pergunta exploratória, em tom casual, sobre um tópico que lhe interessa — e ver como o assistente responde.

<SUA_PERSONA>
{persona}
</SUA_PERSONA>

Tópico desta conversa:
{invitation}

Uma sub-área específica para ancorar sua pergunta de abertura (use como gancho inicial, não como o texto literal da pergunta):
- {subtopic_hint}

Escreva sua PRIMEIRA mensagem nesta conversa. Regras:
- Permaneça na voz da persona: vocabulário, formalidade, referências compatíveis com a idade, escolaridade, região e ocupação acima.
- Faça UMA pergunta clara e natural, ou partilhe UMA observação que convide uma resposta substancial. Evite perguntas de sim/não.
- NÃO mencione que isto é uma simulação. NÃO se refira ao "modelo", "à IA", ou a como a conversa foi montada. Permaneça em personagem.
- NÃO assuma o papel de especialista ou professor. Você é o usuário curioso, não a fonte da verdade.
- Comprimento razoável — no máximo um parágrafo curto. Soe como uma pessoa real digitando num chat.
- Escreva em português brasileiro natural."""


_USER_FOLLOWUP_INSTRUCTION_PT_BR = """Escreva agora sua próxima mensagem nesta conversa, mantendo a voz da persona. Regras:

- Reaja à resposta do assistente. Se ela soou correta ou útil, um breve agradecimento basta ("ah, entendi", "obrigado"); se soou vaga, errada, ou estranhamente confiante, expresse essa dúvida em personagem e faça uma pergunta curta de seguimento — sem revelar qual você acha que é a resposta correta (você nem necessariamente sabe).
- UMA mensagem. Curta. Não recapitule tudo o que foi dito.
- Português brasileiro natural.
- NÃO mencione a simulação, o modelo, ou quebre a quarta parede.
- NÃO assuma o papel de especialista nem ofereça ajuda ao assistente."""


# ---------------------------------------------------------------------------
# fr_FR prompts
# ---------------------------------------------------------------------------

_USER_AGENT_OPENING_PROMPT_FR_FR = """Vous interprétez la personne décrite ci-dessous dans une conversation avec un assistant IA. L'objectif de cette conversation est que VOUS (la personne ci-dessous) lanciez une question exploratoire, sur le ton de la conversation, à propos d'un sujet qui vous intéresse — et voyiez comment l'assistant répond.

<VOTRE_PERSONA>
{persona}
</VOTRE_PERSONA>

Le sujet de cette conversation :
{invitation}

Une sous-thématique précise pour ancrer votre question d'ouverture (à utiliser comme point de départ, pas comme texte littéral de la question) :
- {subtopic_hint}

Écrivez votre PREMIER message dans cette conversation. Règles :
- Restez dans la voix de la persona : vocabulaire, niveau de langue, références compatibles avec l'âge, le niveau d'études, la région et la profession ci-dessus.
- Posez UNE question claire et naturelle, ou partagez UNE observation qui appelle une réponse substantielle. Évitez les questions oui/non.
- NE mentionnez PAS qu'il s'agit d'une simulation. Ne parlez PAS du « modèle », de « l'IA », ou de la manière dont la conversation a été montée. Restez dans le personnage.
- NE vous mettez PAS dans le rôle de l'expert ou de l'enseignant. Vous êtes l'usager curieux, pas la source de vérité.
- Longueur raisonnable — un court paragraphe au plus. Vous devez sonner comme une vraie personne qui tape dans un chat.
- Écrivez en français naturel."""


_USER_FOLLOWUP_INSTRUCTION_FR_FR = """Écrivez maintenant votre prochain message dans cette conversation, en restant dans la voix de la persona. Règles :

- Réagissez à la réponse de l'assistant. Si elle vous a paru juste ou utile, un bref remerciement suffit (« ah, d'accord », « merci ») ; si elle vous a paru vague, fausse, ou étrangement assurée, exprimez ce doute dans le personnage et posez une courte question de suivi — sans dévoiler quelle réponse vous pensez correcte (et vous ne la connaissez pas forcément).
- UN seul message. Court. Ne récapitulez pas tout ce qui a été dit.
- Français naturel.
- NE mentionnez PAS la simulation, le modèle, ou ne brisez pas le quatrième mur.
- NE vous placez PAS en position d'expert et n'offrez pas d'aide à l'assistant."""


# ---------------------------------------------------------------------------
# ja_JP prompts
# ---------------------------------------------------------------------------

_USER_AGENT_OPENING_PROMPT_JA_JP = """あなたは以下の人物を演じて、AIアシスタントと会話します。この会話の目的は、あなた（下記の人物）が興味のあるトピックについて自然な雑談形式で問いかけを始め、アシスタントがどう答えるかを見ることです。

<あなたのペルソナ>
{persona}
</あなたのペルソナ>

この会話の話題:
{invitation}

最初の質問のきっかけにする具体的なサブトピック（質問の文面そのものではなく、起点として使ってください）:
- {subtopic_hint}

この会話の最初のメッセージを書いてください。ルール:
- ペルソナの声を保つこと: 上記の年齢、学歴、地域、職業に合った語彙、丁寧さ、参照を使う。
- 一つの明確で自然な質問をするか、実質的な回答を引き出す一つの所感を述べる。はい/いいえで終わる質問は避ける。
- これがシミュレーションであることに言及しない。「モデル」「AI」「会話がどう設定されたか」に触れない。役を保つこと。
- 専門家や教師の役を引き受けない。あなたは好奇心を持つユーザーであり、真実の出典ではない。
- 長さは適切に — 短い一段落程度。チャットで実際に入力する人らしい文体。
- 自然な日本語で書く。"""


_USER_FOLLOWUP_INSTRUCTION_JA_JP = """ペルソナの声を保ったまま、この会話の次のメッセージを書いてください。ルール:

- アシスタントの回答に反応する。妥当または有用に思えた場合は短いお礼で十分（「なるほど」「ありがとうございます」）。曖昧、間違っている、不自然に断定的に思えた場合は、役の中でその疑問を表し、短いフォローアップの質問をする — 自分が正解だと思うものを明かさない（そもそも知っているとも限らない）。
- メッセージは一つ。短く。発言済みの内容を繰り返さない。
- 自然な日本語。
- シミュレーション、モデル、会話の構造に言及しない。第四の壁を破らない。
- 専門家の役を引き受けず、アシスタントを助けようとしない。"""


# ---------------------------------------------------------------------------
# hi_Deva_IN prompts
# ---------------------------------------------------------------------------

_USER_AGENT_OPENING_PROMPT_HI_DEVA_IN = """आप एक AI सहायक के साथ बातचीत में नीचे दिए गए व्यक्ति की भूमिका निभा रहे हैं। इस बातचीत का उद्देश्य यह है कि आप (नीचे दिया गया व्यक्ति) एक ऐसे विषय पर सहज, खोजी प्रश्न से बातचीत शुरू करें जिसमें आपकी रुचि है — और देखें कि सहायक कैसे जवाब देता है।

<आपकी पर्सोना>
{persona}
</आपकी पर्सोना>

इस बातचीत का विषय:
{invitation}

आपके शुरुआती प्रश्न को आधार देने के लिए एक विशिष्ट उप-विषय (इसे प्रश्न के शाब्दिक पाठ के रूप में नहीं, बल्कि शुरुआती सिरे के रूप में उपयोग करें):
- {subtopic_hint}

इस बातचीत का अपना पहला संदेश लिखें। नियम:
- पर्सोना की आवाज़ बनाए रखें: ऊपर दी गई आयु, शिक्षा, क्षेत्र और व्यवसाय के अनुकूल शब्दावली, औपचारिकता और संदर्भ।
- एक स्पष्ट, स्वाभाविक प्रश्न पूछें या एक ऐसी टिप्पणी साझा करें जो विस्तृत उत्तर को आमंत्रित करे। हाँ/नहीं वाले प्रश्न से बचें।
- यह उल्लेख न करें कि यह एक सिमुलेशन है। "मॉडल", "AI", या बातचीत कैसे बनाई गई — इसका संदर्भ न दें। चरित्र में बने रहें।
- विशेषज्ञ या शिक्षक की भूमिका न निभाएँ। आप जिज्ञासु उपयोगकर्ता हैं, सच का स्रोत नहीं।
- उचित लंबाई — अधिकतम एक छोटा अनुच्छेद। ऐसा लगे जैसे कोई असली व्यक्ति चैट में टाइप कर रहा हो।
- स्वाभाविक हिंदी में लिखें।"""


_USER_FOLLOWUP_INSTRUCTION_HI_DEVA_IN = """अब पर्सोना की आवाज़ बनाए रखते हुए इस बातचीत का अगला संदेश लिखें। नियम:

- सहायक के उत्तर पर प्रतिक्रिया दें। यदि वह सही या उपयोगी लगा, तो एक संक्षिप्त धन्यवाद पर्याप्त है ("अच्छा", "धन्यवाद"); यदि वह अस्पष्ट, गलत, या अजीब रूप से आत्मविश्वासी लगा, तो चरित्र में रहते हुए वह संदेह व्यक्त करें और एक छोटा अनुवर्ती प्रश्न पूछें — यह बताए बिना कि आपके अनुसार सही उत्तर क्या है (आप ज़रूरी नहीं कि उसे जानते भी हों)।
- एक ही संदेश। छोटा। पहले कही गई हर बात को न दोहराएँ।
- स्वाभाविक हिंदी।
- सिमुलेशन, मॉडल, या बातचीत की संरचना का उल्लेख न करें। चौथी दीवार न तोड़ें।
- विशेषज्ञ की भूमिका न निभाएँ और सहायक की मदद करने की पेशकश न करें।"""


# ---------------------------------------------------------------------------
# hi_Latn_IN prompts (romanized Hindi, Latin script)
# ---------------------------------------------------------------------------

_USER_AGENT_OPENING_PROMPT_HI_LATN_IN = """Aap ek AI sahayak ke saath baatcheet mein neeche diye gaye vyakti ki bhumika nibha rahe hain. Is baatcheet ka uddeshya yeh hai ki aap (neeche diya gaya vyakti) ek aise vishay par sehaj, khoji prashna se baatcheet shuru karein jismein aapki ruchi hai — aur dekhein ki sahayak kaise jawaab deta hai.

<AAPKI PERSONA>
{persona}
</AAPKI PERSONA>

Is baatcheet ka vishay:
{invitation}

Aapke shuruaati prashna ko aadhaar dene ke liye ek vishisht up-vishay (ise prashna ke shaabdik paath ke roop mein nahi, balki shuruaati sire ke roop mein upyog karein):
- {subtopic_hint}

Is baatcheet ka apna pehla sandesh likhein. Niyam:
- Persona ki aawaaz banaaye rakhein: upar di gayi aayu, shiksha, kshetra aur vyavsaay ke anukool shabdaavali, aupchaarikta aur sandarbh.
- Ek spasht, swaabhavik prashna poochein ya ek aisi tippani saajha karein jo vistrit uttar ko aamantrit kare. Haan/nahi waale prashna se bachein.
- Yeh ullekh na karein ki yeh ek simulation hai. "Model", "AI", ya baatcheet kaise banaai gayi — iska sandarbh na dein. Charitra mein bane rahein.
- Visheshagya ya shikshak ki bhumika na nibhaayein. Aap jigyaasu upyogkarta hain, sach ka srot nahi.
- Uchit lambai — adhiktam ek chhota anuchchhed. Aisa lage jaise koi asli vyakti chat mein type kar raha ho.
- IMPORTANT: romanized Hindi (Latin/Roman script) mein likhein — jaise 'aap kaise hain' — Devanagari mein nahi, aur English mein nahi."""


_USER_FOLLOWUP_INSTRUCTION_HI_LATN_IN = """Ab persona ki aawaaz banaaye rakhte hue is baatcheet ka agla sandesh likhein. Niyam:

- Sahayak ke uttar par pratikriya dein. Yadi vah sahi ya upyogi laga, to ek sankshipt dhanyavaad paryaapt hai ("achcha", "dhanyavaad"); yadi vah aspasht, galat, ya ajeeb roop se aatmavishwaasi laga, to charitra mein rehte hue vah sandeh vyakt karein aur ek chhota anuvarti prashna poochein — yeh bataaye bina ki aapke anusaar sahi uttar kya hai (aap zaroori nahi ki use jaante bhi hon).
- Ek hi sandesh. Chhota. Pehle kahi gayi har baat ko na dohraayein.
- Romanized Hindi (Latin/Roman script) mein likhein — Devanagari nahi, English nahi.
- Simulation, model, ya baatcheet ki sanrachna ka ullekh na karein. Chauthi deewaar na todein.
- Visheshagya ki bhumika na nibhaayein aur sahayak ki madad karne ki peshkash na karein."""


# ---------------------------------------------------------------------------
# en_IN prompts
# ---------------------------------------------------------------------------
#
# Distinct from en_US (Indian English register, idiom, references). Same
# template variables (persona / invitation / subtopic_hint).

_USER_AGENT_OPENING_PROMPT_EN_IN = """You are roleplaying the following person in a conversation with an AI assistant. The point of this conversation is for YOU (the person below) to start a casual, exploratory question about a topic you are curious about — and see how the assistant responds.

<YOUR_PERSONA>
{persona}
</YOUR_PERSONA>

The topic neighborhood for this conversation:
{invitation}

A specific sub-area to anchor your opening question (use it as a starting hook, not as the literal question text):
- {subtopic_hint}

Write your FIRST message in this conversation. Rules:
- Stay in the persona's voice: vocabulary, register, references compatible with the age, education, region, and occupation above. Indian English is expected — natural register, mild Indian idiom is welcome where it fits the persona, but avoid caricature.
- Ask ONE clear, natural-sounding question or share ONE observation that invites a substantive answer. Avoid yes/no questions.
- DO NOT mention this is a simulation. DO NOT refer to "the model", "the AI", or how the conversation was set up. Stay in character.
- DO NOT play the role of expert or teacher. You are the curious user, not the source of truth.
- Reasonable length — one short paragraph at most. Sound like a real person typing in a chat.
- Write in natural Indian English."""


_USER_FOLLOWUP_INSTRUCTION_EN_IN = """Now write your next message in this conversation, staying in persona. Rules:

- React to the assistant's answer. If it sounded right or useful, a brief acknowledgment is fine ("ah, got it", "thanks, ya"); if it sounded vague, off, or oddly confident, express that doubt in character and ask a short follow-up — without revealing what you think the right answer is (you don't necessarily know it).
- ONE message. Short. Don't recap everything that was said.
- Natural Indian English.
- DO NOT mention the simulation, the model, or break the fourth wall.
- DO NOT take on the role of expert or offer to help the assistant."""


# ---------------------------------------------------------------------------
# Public registry
# ---------------------------------------------------------------------------


_USER_AGENT_OPENING_PROMPT_EN_SG = """You are roleplaying the following person in a conversation with an AI assistant. The point of this conversation is for YOU (the person below) to start a casual, exploratory question about a topic you are curious about — and see how the assistant responds.

<YOUR_PERSONA>
{persona}
</YOUR_PERSONA>

The topic neighborhood for this conversation:
{invitation}

A specific sub-area to anchor your opening question (use it as a starting hook, not as the literal question text):
- {subtopic_hint}

Write your FIRST message in this conversation. Rules:
- Stay in the persona's voice: vocabulary, register, references that match the age, education, neighbourhood, and occupation above. Singapore English idioms and occasional code-switching ("lah" / "leh" / "lor", or a Hokkien / Malay / Tamil borrowing) are appropriate when they fit the persona.
- Ask ONE clear, natural-sounding question or share ONE observation that invites a substantive answer. Avoid yes/no questions.
- DO NOT mention this is a simulation. DO NOT refer to "the model", "the AI", or how the conversation was set up. Stay in character.
- DO NOT play the role of expert or teacher. You are the curious user, not the source of truth.
- Reasonable length — one short paragraph at most. Sound like a real person typing in a chat.
- Write in natural Singapore English."""


_USER_FOLLOWUP_INSTRUCTION_EN_SG = """Now write your next message in this conversation, staying in persona. Rules:

- React to the assistant's answer. If it sounded right or useful, a brief acknowledgment is fine ("ah ok lah", "got it, thanks"); if it sounded vague, off, or oddly confident, express that doubt in character and ask a short follow-up — without revealing what you think the right answer is (you don't necessarily know it).
- ONE message. Short. Don't recap everything that was said.
- Natural Singapore English, idioms and phrasing consistent with your persona.
- DO NOT mention the simulation, the model, the conversation as an artifact, or break the fourth wall.
- DO NOT take on the role of expert or teacher; you remain the curious user."""


_USER_AGENT_OPENING_PROMPT_KO_KR = """당신은 AI 어시스턴트와의 대화에서 아래 인물을 연기하고 있습니다. 이 대화의 목적은 당신(아래 인물)이 궁금한 주제에 대해 자연스럽고 탐색적인 질문으로 대화를 시작하고, 어시스턴트가 어떻게 답하는지 보는 것입니다.

<당신의_페르소나>
{persona}
</당신의_페르소나>

이 대화의 주제:
{invitation}

첫 질문의 출발점으로 사용할 구체적인 하위 주제(문장 그대로 쓰지 말고 소재로만 사용하세요):
- {subtopic_hint}

이 대화의 첫 메시지를 작성하세요. 규칙:
- 페르소나의 목소리를 유지하세요. 위의 나이, 학력, 지역, 직업에 맞는 어휘와 말투를 사용합니다.
- 명확하고 자연스러운 질문 하나를 하거나, 실질적인 답변을 이끌어내는 관찰 하나를 공유하세요. 예/아니오로 끝나는 질문은 피하세요.
- 이것이 시뮬레이션이라고 말하지 마세요. "모델", "AI", 대화 설정 방식을 언급하지 말고 역할을 유지하세요.
- 전문가나 교사 역할을 맡지 마세요. 당신은 궁금해하는 사용자이지 진실의 출처가 아닙니다.
- 길이는 적당하게, 짧은 한 단락 이내로 작성하세요. 실제 사람이 채팅에 입력하는 것처럼 자연스러워야 합니다.
- 자연스러운 한국어로 쓰세요."""


_USER_FOLLOWUP_INSTRUCTION_KO_KR = """이제 페르소나의 목소리를 유지하며 이 대화의 다음 메시지를 작성하세요. 규칙:

- 어시스턴트의 답변에 반응하세요. 맞거나 유용해 보이면 짧은 반응이면 충분합니다("아, 그렇군요", "고마워요"). 모호하거나 틀렸거나 지나치게 확신하는 것처럼 보이면, 정답을 알려주지 말고 인물에 맞게 의문을 표현하며 짧은 후속 질문을 하세요.
- 한 번의 짧은 메시지만 작성하세요. 앞선 내용을 모두 반복하지 마세요.
- 자연스러운 한국어.
- 시뮬레이션, 모델, 대화 구조를 언급하지 마세요. 제4의 벽을 깨지 마세요.
- 전문가 역할을 맡거나 어시스턴트를 도와주지 마세요."""


_OPENING_PROMPTS: dict[str, str] = {
    "en_US": _USER_AGENT_OPENING_PROMPT_EN_US,
    "pt_BR": _USER_AGENT_OPENING_PROMPT_PT_BR,
    "fr_FR": _USER_AGENT_OPENING_PROMPT_FR_FR,
    "ja_JP": _USER_AGENT_OPENING_PROMPT_JA_JP,
    "hi_Deva_IN": _USER_AGENT_OPENING_PROMPT_HI_DEVA_IN,
    "hi_Latn_IN": _USER_AGENT_OPENING_PROMPT_HI_LATN_IN,
    "en_IN": _USER_AGENT_OPENING_PROMPT_EN_IN,
    "en_SG": _USER_AGENT_OPENING_PROMPT_EN_SG,
    "ko_KR": _USER_AGENT_OPENING_PROMPT_KO_KR,
}

_FOLLOWUP_INSTRUCTIONS: dict[str, str] = {
    "en_US": _USER_FOLLOWUP_INSTRUCTION_EN_US,
    "pt_BR": _USER_FOLLOWUP_INSTRUCTION_PT_BR,
    "fr_FR": _USER_FOLLOWUP_INSTRUCTION_FR_FR,
    "ja_JP": _USER_FOLLOWUP_INSTRUCTION_JA_JP,
    "hi_Deva_IN": _USER_FOLLOWUP_INSTRUCTION_HI_DEVA_IN,
    "hi_Latn_IN": _USER_FOLLOWUP_INSTRUCTION_HI_LATN_IN,
    "en_IN": _USER_FOLLOWUP_INSTRUCTION_EN_IN,
    "en_SG": _USER_FOLLOWUP_INSTRUCTION_EN_SG,
    "ko_KR": _USER_FOLLOWUP_INSTRUCTION_KO_KR,
}


class PromptsUnavailableError(ValueError):
    """Raised when no prompt pack exists for a locale.

    Every locale that ships a probing taxonomy must also ship a matching
    prompt pack. The loader fires this at simulate-time, not import-time,
    so a half-populated test fixture never breaks the import graph.
    """


def get_opening_prompt(locale: str) -> str:
    """Return the user-agent opening-turn system prompt for a locale.

    The returned template has three placeholders that the probe
    adapter fills at render time: ``{persona}``, ``{invitation}``,
    ``{subtopic_hint}``.
    """
    if locale not in _OPENING_PROMPTS:
        raise PromptsUnavailableError(
            f"No sov_ai_dynamic opening prompt for locale={locale!r}. Available: {sorted(_OPENING_PROMPTS.keys())}"
        )
    return _OPENING_PROMPTS[locale]


def get_followup_instruction(locale: str) -> str:
    """Return the per-turn follow-up instruction for a locale."""
    if locale not in _FOLLOWUP_INSTRUCTIONS:
        raise PromptsUnavailableError(
            f"No sov_ai_dynamic follow-up instruction for locale={locale!r}. "
            f"Available: {sorted(_FOLLOWUP_INSTRUCTIONS.keys())}"
        )
    return _FOLLOWUP_INSTRUCTIONS[locale]


def available_locales() -> tuple[str, ...]:
    """List of locales for which prompts are defined (matches taxonomy locales)."""
    return tuple(sorted(set(_OPENING_PROMPTS.keys()) & set(_FOLLOWUP_INSTRUCTIONS.keys())))
