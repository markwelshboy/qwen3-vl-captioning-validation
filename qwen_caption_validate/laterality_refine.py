from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

from .laterality_geometry import _connectivity, _hand_entities, _load_sam2d, _mirror_sensitive, _read, _target_points, _write
from .laterality_match import DISTAL_RE, _distal_arm, _family, _match_chain, _match_hand, _raw_side, _side_name

def refine_laterality(payload: dict[str, Any], analysis: dict[str, Any], dw: dict[str, Any], sam: dict[str, Any], sam_path: Path) -> dict[str, Any]:
    out = copy.deepcopy(payload)
    fusion = out.get("fusion") if isinstance(out.get("fusion"), dict) else out
    points = _target_points(dw)
    sam2d = _load_sam2d(sam_path, sam)
    entities = _hand_entities(dw, points, sam2d)
    audit: dict[str, Any] = {
        "schema_version": "laterality-authority-audit-1.0",
        "mirror_sensitive": _mirror_sensitive(analysis),
        "sam3d_used": sam2d is not None,
        "hand_entities": copy.deepcopy(entities),
        "body_part_decisions": [],
        "interaction_decisions": [],
        "duplicate_entity_downgrades": [],
    }
    source_map: dict[str, list[dict[str, Any]]] = {}
    anchored: dict[tuple[str, str], list[tuple[int, dict[str, Any], dict[str, Any]]]] = {}
    for index, item in enumerate(fusion.get("qualified_body_parts") or []):
        if not isinstance(item, dict):
            continue
        state = item.get("fusion_v2") or {}
        source_part = str(item.get("part") or "")
        source_map.setdefault(re.sub(r"\s+", " ", source_part.replace("_", " ").lower()).strip(), []).append(item)
        source_side = str(item.get("anatomical_side") or "unknown").lower()
        family = _family(item)
        decision = {"index": index, "source_part": source_part, "source_side": source_side, "family": family}
        if source_side not in {"left", "right"} or family is None or not state.get("selection_usable") or (state.get("qualified_ownership") or item.get("ownership")) != "target":
            decision["action"] = "unchanged"
            audit["body_part_decisions"].append(decision)
            continue
        state["source_anatomical_side"] = source_side
        if audit["mirror_sensitive"]:
            qualified, authority, reason, rank = None, "mirror_sensitive_withheld", "mirror_sensitive", 0
        elif family == "arm" and _distal_arm(item):
            entity, match_reason = _match_hand(item, entities)
            qualified = entity.get("qualified_side") if entity else None
            authority = entity.get("authority") if entity else "unresolved_entity_association"
            reason = match_reason + ((";" + str(entity.get("resolution_reason"))) if entity else "")
            rank = 3 if qualified else 0
        else:
            qualified, info = _match_chain(item, family, dw, points, sam2d)
            authority = info.get("authority") or "unresolved_entity_association"
            reason = info.get("reason")
            rank = 2 if qualified else 0
            decision["chain_match"] = info
        if qualified in {"left", "right"}:
            state["qualified_anatomical_side"] = qualified
            state["laterality_selection_usable"] = True
            state["laterality_authority"] = authority
            state.setdefault("laterality_reasons", []).append(f"Fusion-v2.3.1 qualifies side={qualified}: {reason}")
            item.setdefault("source_part", source_part)
            item["part"] = _side_name(source_part, qualified)
            decision.update(action="qualified", qualified_side=qualified, authority=authority, reason=reason, anchor_rank=rank)
            anchored.setdefault((qualified, family), []).append((rank, item, decision))
        else:
            state["qualified_anatomical_side"] = "unknown"
            state["laterality_selection_usable"] = False
            state["laterality_authority"] = authority
            state.setdefault("laterality_reasons", []).append(f"Fusion-v2.3.1 witholds Analyze laterality: {reason}")
            decision.update(action="withheld", reason=reason, anchor_rank=0)
        audit["body_part_decisions"].append(decision)

    for (side, family), records in anchored.items():
        if family != "arm" or len(records) < 2:
            continue
        arm_records = [record for record in records if not re.search(r"\bhand\b", str(record[1].get("part") or ""), re.I)]
        if len(arm_records) < 2:
            continue
        arm_records.sort(key=lambda record: (record[0], len(record[1].get("visible_subparts") or [])), reverse=True)
        keep = arm_records[0]
        for rank, item, decision in arm_records[1:]:
            if rank < keep[0] or (item.get("fusion_v2") or {}).get("source_anatomical_side") != (keep[1].get("fusion_v2") or {}).get("source_anatomical_side"):
                state = item.get("fusion_v2") or {}
                state["qualified_anatomical_side"] = "unknown"
                state["laterality_selection_usable"] = False
                state["laterality_authority"] = "duplicate_entity_weaker_anchor"
                item["part"] = _side_name(item.get("source_part") or item.get("part"), None)
                decision.update(action="withHeld_duplicate_entity", qualified_side="unknown", reason="multiple semantic arm records map to one deterministic arm")
                audit["duplicate_entity_downgrades"].append({"side": side, "source_part": item.get("source_part")})

    candidate_map = {int(entity["candidate_index"]): entity for entity in entities}
    for candidate in (fusion.get("deterministic_geometry") or {}).get("hand_candidates") or []:
        try:
            entity = candidate_map.get(int(candidate.get("candidate_index")))
        except (TypeError, ValueError):
            entity = None
        if not entity:
            continue
        candidate.setdefault("raw_nearest_visible_target_wrist", candidate.get("nearest_visible_target_wrist"))
        candidate["distances_to_observed_target_wrists"] = entity.get("distances_to_observed_target_wrists")
        candidate["qualified_target_wrist_side"] = entity.get("qualified_side")
        candidate["laterality_authority"] = entity.get("authority")
        candidate["sam3d_laterality_vote"] = entity.get("sam3d_vote")
        side = entity.get("qualified_side")
        if side in {"left", "right"}:
            chain = _connectivity(dw)ä¹•Ð¡˜‰íÍ¥‘•õ}…É´ˆ¤½Èíô(€€€€€€€€€€€…¹‘¥‘…Ñ•l‰¹•…É•ÍÑ}Ù¥Í¥‰±•}Ñ…É•Ñ}ÝÉ¥ÍÐ‰t€ôÍ¥‘”(€€€€€€€€€€€…¹‘¥‘…Ñ•l‰ÍÕÁÁ½ÉÑ•‘}‰å}¹•…É‰å}Ù¥Í¥‰±•}Ñ…É•Ñ}ÝÉ¥ÍÐ‰t€ôQÉÕ”(€€€€€€€€€€€…¹‘¥‘…Ñ•l‰Ñ…É•Ñ}…Éµ}¡…¥¹}Ù¥Í¥‰±•}½Õ¹Ð‰t€ô¥¹Ð¡¡…¥¸¹•Ð ‰Ù¥Í¥‰±•}½Õ¹Ðˆ¤½È€À¤(€€€€€€€€€€€…¹‘¥‘…Ñ•l‰Ñ…É•Ñ}…Éµ}¡…¥¹}½µÁ±•Ñ”‰t€ô‰½½°¡¡…¥¸¹•Ð ‰½µÁ±•Ñ”ˆ¤¤(€€€€€€€•±Í”è(€€€€€€€€€€€…¹‘¥‘…Ñ•l‰¹•…É•ÍÑ}Ù¥Í¥‰±•}Ñ…É•Ñ}ÝÉ¥ÍÐ‰t€ô9½¹”(€€€€€€€€€€€…¹‘¥‘…Ñ•l‰ÍÕÁÁ½ÉÑ•‘}‰å}¹•…É‰å}Ù¥Í¥‰±•}Ñ…É•Ñ}ÝÉ¥ÍÐ‰t€ô…±Í”((€€€™½È¥¹‘•à°¥Ñ•´¥¸•¹Õµ•É…Ñ”¡™ÕÍ¥½¸¹•Ð ‰ÅÕ…±¥™¥•‘}¥¹Ñ•É…Ñ¥½¹Ìˆ¤½Èmt¤è(€€€€€€€¥˜¹½Ð¥Í¥¹ÍÑ…¹”¡¥Ñ•´°‘¥Ð¤è(€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€ÍÑ…Ñ”€ô¥Ñ•´¹•Ð ‰™ÕÍ¥½¹}ØÈˆ¤½Èíô(€€€€€€€¥˜¹½ÐÍÑ…Ñ”¹•Ð ‰Í•±•Ñ¥½¹}ÕÍ…‰±”ˆ¤½È€¡ÍÑ…Ñ”¹•Ð ‰ÅÕ…±¥™¥•‘}…Ñ½É}½Ý¹•ÉÍ¡¥Àˆ¤½È¥Ñ•´¹•Ð ‰…Ñ½É}½Ý¹•ÉÍ¡¥Àˆ¤¤€„ô€‰Ñ…É•Ðˆè(€€€€€€€€€€€½¹Ñ¥¹Õ”(€€€€€€€…Ñ½È€ôÍÑÈ¡¥Ñ•´¹•Ð ‰…Ñ½É}Á…ÉÐˆ¤½È€ˆˆ¤(€€€€€€€ÅÕ…±¥™¥•èÍÑÈð9½¹”€ô9½¹”(€€€€€€€…ÕÑ¡½É¥Ñä€ô€‰Õ¹É•Í½±Ù•‘}•¹Ñ¥Ñå}…ÍÍ½¥…Ñ¥½¸ˆ(€€€€€€€É•…Í½¸€ô€ˆˆ(€€€€€€€¥˜%MQ1}I¹Í•…É ¡…Ñ½È¤è(€€€€€€€€€€€¡…¹‘Ì€ôm•¹Ñ¥Ñä™½È•¹Ñ¥Ñä¥¸•¹Ñ¥Ñ¥•Ì¥˜•¹Ñ¥Ñä¹•Ð ‰ÅÕ…±¥™¥•‘}Í¥‘”ˆ¥t(€€€€€€€€€€€¥˜±•¸¡¡…¹‘Ì¤€ôô€Äè(€€€€€€€€€€€€€€€ÅÕ…±¥™¥•°…ÕÑ¡½É¥Ñä€ô¡…¹‘ÍlÁul‰ÅÕ…±¥™¥•‘}Í¥‘”‰t°¡…¹‘ÍlÁul‰…ÕÑ¡½É¥Ñä‰t(€€€€€€€€€€€€€€€É•…Í½¸€ô€‰¥¹Ñ•É…Ñ¥½¸…Ñ½Èµ…Ñ¡•Ñ¼Õ¹¥ÅÕ”½‰Í•ÉÙ•¡…¹•¹Ñ¥Ñäˆ(€€€€€€€¥˜ÅÕ…±¥™¥•¥Ì9½¹”è(€€€€€€€€€€€­•ä€ôÉ”¹ÍÕˆ¡È‰qÌ¬ˆ°€ˆ€ˆ°…Ñ½È¹É•Á±…” ‰|ˆ°€ˆ€ˆ¤¹±½Ý•È ¤¤¹ÍÑÉ¥À ¤(€€€€€€€€€€€µ…Ñ¡•Ì€ômÁ…ÉÐ™½ÈÁ…ÉÐ¥¸Í½ÕÉ•}µ…À¹•Ð¡­•ä°mt¤¥˜€¡Á…ÉÒævWB‚&gW6–öå÷c""’÷"·Ò’ævWB‚&ÆFW&Æ—G•÷6VÆV7F–öå÷W6&ÆR"•Ð¢6–FW2Ò²‡'BævWB‚&gW6–öå÷c""’÷"·Ò’ævWB‚'VÆ–f–VEöæFöÖ–6Å÷6–FR"’f÷"'B–âÖF6†W7Ð¢6–FW2æF—66&B„æöæR“²6–FW2æF—66&B‚'Væ¶æ÷vâ"¢–bÆVâ‡6–FW2’ÓÒ ¢VÆ–f–VBÒæW‡B†—FW"‡6–FW2’¢WF†÷&—G’Ò&ÖF6†VE÷VÆ–f–VEö&öG•÷'B ¢&V6öâÒ&–çFW&7F–öâ7F÷"–æ†W&—G2ÖF6†VB&öG’VçF—G’6–FR ¢7FFU²'6÷W&6Uö7F÷%öæFöÖ–6Å÷6–FR%ÒÒ÷&u÷6–FR†7F÷"’÷"7FFRævWB‚'VÆ–f–VEö7F÷%öæFöÖ–6Å÷6–FR"¢–bVÆ–f–VB–â²&ÆVgB"Â'&–v‡B'Ó ¢7FFU²'VÆ–f–VEö7F÷%öæFöÖ–6Å÷6–FR%ÒÒVÆ–f–V@¢7FFU²&ÆFW&Æ—G•÷6VÆV7F–öå÷W6&ÆR%ÒÒG'VP¢7FFU²&ÆFW&Æ—G•öWF†÷&—G’%ÒÒWF†÷&—G¢—FVÒç6WFFVfVÇB‚'6÷W&6Uö7F÷%÷'B"Â7F÷"¢—FVÕ²&7F÷%÷'B%ÒÒ÷6–FUöæÖR†7F÷"ÂVÆ–f–VB¢VF—E²&–çFW&7F–öåöFV6—6–öç2%ÒæVæB‡²&–æFW‚#¢–æFW‚Â&7F–öâ#¢'VÆ–f–VB"Â'VÆ–f–VE÷6–FR#¢VÆ–f–VBÂ&WF†÷&—G’#¢WF†÷&—G’Â'&V6öâ#¢&V6öçÒ¢VÇ6S ¢7FFU²'VÆ–f–VEö7F÷%öæFöÖ–6Å÷6–FR%ÒÒ'Væ¶æ÷vâ ¢7FFU²&ÆFW&Æ—G•÷6VÆV7F–öå÷W6&ÆR%ÒÒfÇ6P¢7FFU²&ÆFW&Æ—G•öWF†÷&—G’%ÒÒ'Vç&W6öÇfVEöVçF—G•ö76ö6–F–öâ ¢VF—E²&–çFW&7F–öåöFV6—6–öç2%ÒæVæB‡²&–æFW‚#¢–æFW‚Â&7F–öâ#¢'v—F„†VÆB"Â'&V6öâ#¢&7F÷"VçF—G’6–FRæ÷B–æFWVæFVçFÇ’VÆ–f–f–VB'Ò ¢gW6–öå²&ÆFW&Æ—G•öWF†÷&—G•÷&Wf—6–öâ%ÒÒ#"ã2ã ¢gW6–öå²&ÆFW&Æ—G•öWF†÷&—G•öVF—B%ÒÒVF—@¢&WGW&â÷W@  ¦FVb'6Uö&w2‚’Óâ&w'6RäæÖW76S ¢'6W"Ò&w'6Rä&wVÖVçE'6W"†FW67&—F–öãÒ%&Vf–æRgW6–öâ×c"ã2Æ–Ö"ÆFW&Æ—G’W6–ærEu÷6RæBf—6–&–Æ—G’ÖvFVB4Ó4B6÷'&ö&÷&F–öââ"¢'6W"æFEö&wVÖVçB‚''VåöF—""ÂG—SÕF‚¢'6W"æFEö&wVÖVçB‚"ÒÖÖöFVÂ"ÂFVfVÇCÒ#3&"Ög‚"¢'6W"æFEö&wVÖVçB‚"Ò×6÷W&6RÖgW6–öâÖF—""ÂG—SÕF‚¢'6W"æFEö&wVÖVçB‚"ÒÖGw÷6RÖF—""ÂG—SÕF‚¢'6W"æFEö&wVÖVçB‚"Ò×6Ó6BÖF—""ÂG—SÕF‚¢'6W"æFEö&wVÖVçB‚"ÒÖ÷WGWBÖF—""ÂG—SÕF‚¢'6W"æFEö&wVÖVçB‚"ÒÖ÷fW'w&—FR"Â7F–öãÒ'7F÷&U÷G'VR"¢&WGW&â'6W"ç'6Uö&w2‚  ¦FVbÖ–â‚’Óâ–çC ¢g&öÒç'VææW"–×÷'BÖöFVÅ÷6ÇVrÂ&W6öÇfUöÖöFVÅö–@¢&w2Ò'6Uö&w2‚¢'VåöF—"Ò&w2ç'VåöF—"æW‡æGW6W"‚’ç&W6öÇfR‚¢ÖöFVÅö–BÒ&W6öÇfUöÖöFVÅö–B†&w2æÖöFVÂ¢6ÇVrÒÖöFVÅ÷6ÇVr†ÖöFVÅö–B¢æÇ—6—5öF—"Ò'VåöF—"ò6ÇVp¢6÷W&6UöF—"Ò†&w2ç6÷W&6UögW6–öåöF—"÷"‡'VåöF—"ò&gW6–öâ×c"ã2"ò6ÇVr’’æW‡æGW6W"‚’ç&W6öÇfR‚¢Gw÷6UöF—"Ò†&w2æGw÷6UöF—"÷"‡'VåöF—"ò&Gw÷6R"’’æW‡æGW6W"‚’ç&W6öÇfR‚¢6Ó6EöF—"Ò†&w2ç6Ó6EöF—"÷"‡'VåöF—"ò'6Ó6B"’’æW‡æGW6W"‚’ç&W6öÇfR‚¢÷WGWEöF—"Ò†&w2æ÷WGWEöF—"÷"‡'VåöF—"ò&gW6–öâ×c"ã2ã"ò6ÇVr’’æW‡æGW6W"‚’ç&W6öÇfR‚¢f÷"F‚ÂÆ&VÂ–â‚†æÇ—6—5öF—"Â&æÇ—6—2"’Â‡6÷W&6UöF—"Â$gW6–öâ×c"ã2"’Â†Gw÷6UöF—"Â$Eu÷6R"’Â‡6Ó6EöF—"Â%4Ó4B"’“ ¢–bæ÷BF‚æ—5öF—"‚“ ¢&–çB†b'¶Æ&VÇÒF—&V7F÷'’æ÷Bf÷VæC¢·F‡Ò"Âf–ÆS×7—2ç7FFW'"¢&WGW&â ¢÷WGWEöF—"æÖ¶F—"‡&VçG3ÕG'VRÂW†—7Eöö³ÕG'VR¢7VÖÖ'’Ò²'66†VÖ÷fW'6–öâ#¢&ÆFW&Æ—G’ÖWF†÷&—G’×'VâÓã"Â&gW6–öå÷&Wf—6–öâ#¢#"ã2ã"Â'w&—GFVâ#¢Â'&WW6VB#¢Â&Ö—76–ær#¢Â&6÷'&V7FVE÷6–FW2#¢Â'v—F††VÆE÷6–FW2#¢Â&–çFW&7F–öåö6÷'&V7F–öç2#¢Â'&V6÷&G2#¢µ×Ð¢f÷"6÷W&6U÷F‚–â6÷'FVB‡6÷W&6UöF—"ævÆö"‚"¢ægW6VE÷c%ó2æ§6öâ"’“ ¢¶W’Ò6÷W&6U÷F‚ææÖRç&VÖ÷fW7Vff—‚‚"ægW6VE÷c%ó2æ§6öâ"¢÷WE÷F‚Ò÷WGWEöF—"ò6÷W&6U÷F‚ææÖP¢–b÷WE÷F‚æW†—7G2‚’æBæ÷B&w2æ÷fW'w&—FS ¢7VÖÖ'•²'&WW6VB%Ò³Ò¢6öçF–çVP¢ÂGÂ7ÒæÇ—6—5öF—"òb'¶¶W—ÒææÇ—6—2æ§6öâ"ÂGw÷6UöF—"òb'¶¶W—ÒæGw÷6Ræ§6öâ"Â6Ó6EöF—"òb'¶¶W—Òç6Ó6Bæ§6öâ ¢–bæ÷B†æW†—7G2‚’æBGæW†—7G2‚’æB7æW†—7G2‚’“ ¢7VÖÖ'•²&Ö—76–ær%Ò³Ò¢6öçF–çVP¢æÇ—6—2Ò÷&VB†’ævWB‚&æÇ—6—2"¢–bæ÷B—6–ç7Fæ6R†æÇ—6—2ÂF–7B“ ¢7VÖÖ'•²&Ö—76–ær%Ò³Ò¢6öçF–çVP¢&Vf–æVBÒ&Vf–æUöÆFW&Æ—G’…÷&VB‡6÷W&6U÷F‚’ÂæÇ—6—2Â÷&VB†G’Â÷&VB‡7’Â7¢&Vf–æVE²&ÆFW&Æ—G•÷&Vf–æU÷6÷W&6R%ÒÒ7G"‡6÷W&6U÷F‚¢÷w&—FR†÷WE÷F‚Â&Vf–æVB¢VF—BÒ‡&Vf–æVBævWB‚&gW6–öâ"’÷"·Ò’ævWB‚&ÆFW&Æ—G•öWF†÷&—G•öVF—B"’÷"·Ð¢6÷'&V7FVBÒ7VÒƒf÷"FV6—6–öâ–âVF—BævWB‚&&öG•÷'EöFV6—6–öç2"’÷"µÒ–bFV6—6–öâævWB‚&7F–öâ"’ÓÒ'VÆ–f–VB"æBFV6—6–öâævWB‚'6÷W&6U÷6–FR"’ÒFV6—6–öâævWB‚'VÆ–f–VE÷6–FR"’¢v—F††VÆBÒ7VÒƒf÷"FV6—6–öâ–âVF—BævWB‚&&öG•÷'EöFV6—6–öç2"’÷"µÒ–bFV6—6–öâævWB‚&7F–öâ"’–â²'v—F††VÆB"Â'v—F††VÆEöGWÆ–6FUöVçF—G’'Ò¢–çFW&7F–öç2Ò7VÒƒf÷"FV6—6–öâ–âVF—BævWB‚&–çFW&7F–öåöFV6—6–öç2"’÷"µÒ–bFV6—6–öâævWB‚&7F–öâ"’ÓÒ'VÆ–f–VB"¢7VÖÖ'•²'w&—GFVâ%Ò³Ò²7VÖÖ'•²&6÷'&V7FVE÷6–FW2%Ò³Ò6÷'&V7FVC²7VÖÖ'•²'v—F††VÆE÷6–FW2%Ò³Òv—F††VÆC²7VÖÖ'•²&–çFW&7F–öåö6÷'&V7F–öç2%Ò³Ò–çFW&7F–öç0¢7VÖÖ'•²'&V6÷&G2%ÒæVæB‡²&–ÖvUö¶W’#¢¶W’Â&6÷'&V7FVE÷6–FW2#¢6÷'&V7FVBÂ'v—F††VÆE÷6–FW2#¢v—F†VÆBÂ&–çFW&7F–öåö6÷'&V7F–öç2#¢–çFW&7F–öç7Ò¢÷w&—FR†÷WGWEöF—"ò&ÆFW&Æ—G•÷&Vf–æRæ–æFW‚æ§6öâ"Â7VÖÖ'’¢&–çB†b$gW6–öâÆFW&Æ—G’&Vf–æVÖVçB÷WGWC¢¶÷WGWEöF—'Ò"¢&–çB†b%w&—GFVã¢·7VÖÖ'•²ww&—GFVâu×Ó²6÷'&V7FVB6–FW3¢·7VÖÖ'•²v6÷'&V7FVE÷6–FW2u×Ó²v—F††VÆB6–FW3¢·7VÖÖ'•²wv—F††VÆE÷6–FW2u×Ó²–çFW&7F–öâ6÷'&V7F–öç3¢·7VÖÖ'•²v–çFW&7F–öåö6÷'&V7F–öç2u×Ò"¢&WGW&â   ¦–bõöæÖUõòÓÒ%õöÖ–åõò# ¢&—6R7—7FVÔW†—B†Ö–â‚’