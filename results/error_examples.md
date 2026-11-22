### LSTM + Bahdanau attention

| Category | Hinglish | Model output | Reference |
| --- | --- | --- | --- |
| Junk output | `hamari bus main bhi ek dengu aata tha ,` | `) :p` | `in our bus also one dengu used to come ,` |
| Noisy reference | `guddu to superb actor h . . . chhaa gya yaar` | `guddu is a very good friend` | `) :p` |
| Wrong meaning | `mhanje kaahi pan bolayacha? nustach` | `i am amazing pan pan` | `where do you boil? nustach` |
| Repetition | `coz pappu cant dance saala . . . . gucci ka perfume lagata h sala :p :d` | `i have got it . . . . . . . . . . . . .` | `because pappu can't dance, he sprays on gucci's perfume` |
| Dropped words | `congrats. ab to mere se nahi mangega` | `) now` | `congrats. nnw it will not be asked from me` |
| English not copied | `dis is gujju sparta :prajakta` | `who is far.` | `this is gujju sparta : prajakta` |
| Wrong word choice | `@stupidusmaximus dhanyawaad bhau !!` | `developement takes !!` | `@stupidusmaximus thank you brother !!` |

### Transformer, 6 layers

| Category | Hinglish | Model output | Reference |
| --- | --- | --- | --- |
| Noisy reference | `rt @sheikhimaan : news flash : iss baar bheja tha . jeet gaye .` | `rt @sheikhimaan : news flash : this time has been sent.` | `) <url>` |
| Wrong meaning | `kya yr maine b mis kr diya #uhh` | `what did i do? #uhh` | `damn man, even i missed it` |
| Hindi word left untranslated | `mobarak ied mobarak ??` | `mobarak ied mobarak ??` | `happy eid.` |
| Junk output | `lol....theek hai yaar` | `laughing out loud. this is aek` | `lol! it's okay mate.` |
| Wrong word choice | `kis cheez ke baare mein?` | `whom in this thing?` | `about which one?` |
| Repetition | `yrrr aao na jaldi 4:14 bhi ho gye` | `why should you come to come 4:14 too.` | `friends come fast. it has already been 4:14` |
| Correct but worded differently | `tujhe yaad na meri aayee` | `you don't remember me.` | `you forgot me` |
| Dropped words | `mene phle bhi bola tha . . . ye guddu bigadtaa ja raha hai . . . samjhao koi ese . . ! ! !` | `i said that i also said . . . . guddu is bigadta . . . samjhao` | `i have told you before too . guddu is becoming naughty. someone please make him understand.` |

### mT5-small fine-tuned

| Category | Hinglish | Model output | Reference |
| --- | --- | --- | --- |
| Junk output | `bahot hi galat huwa ya, shraddhanjali arpit karte hai :'(` | `@(` | `what happened was wrong, let us pay homage!` |
| Repetition | `kya yr maine b mis kr diya #uhh` | `@uhhhhhhhhhhh` | `damn man, even i missed it` |
| Noisy reference | `@rofl_india mujhe bhi milwao iss khushi se` | `@rofl_india` | `-p` |
| Dropped words | `mujhe bhi koi book me jagah de do.<url>` | `@url>` | `give me a place in a book too.` |

### mBART-50 fine-tuned

| Category | Hinglish | Model output | Reference |
| --- | --- | --- | --- |
| Noisy reference | `rt @sheikhimaan : news flash : iss baar bheja tha . jeet gaye .` | `rt @sheikhimaan : news flash : sent this time. they won.` | `) <url>` |
| Transliteration error | `mobarak ied mobarak ??` | `mobarak ied mobarak ??` | `happy eid.` |
| Wrong meaning | `bhawnao me beh gya londa :p` | `londa went to sleep :p` | `caught in feelings boy` |
| Wrong word choice | `..... ye jahaan mil gayaa! yes yes!` | `..... where did you find this! yes yes!` | `i got the whole world!` |
| Code-mixing issue | `kya yr maine b mis kr diya #uhh` | `what have i misunderstood #uhh` | `damn man, even i missed it` |
| Dropped words | `@chatpatidiary sachh baat hai bas` | `@chatpatidiary it's true` | `@patttydry is just the truth` |

