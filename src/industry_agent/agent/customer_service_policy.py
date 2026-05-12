"""Lightweight customer-service policy responses for non-manual questions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyResponse:
    answer: str
    confidence: float
    matched_topics: list[str]


@dataclass(frozen=True)
class ScenarioRule:
    name: str
    terms: tuple[str, ...]
    overview: str = ""
    materials: str = ""
    timeline: str = ""
    fees: str = ""
    eligibility: str = ""
    process: str = ""
    contact: str = ""


@dataclass(frozen=True)
class TopicRule:
    topic: str
    terms: tuple[str, ...]
    overview: str
    materials: str
    timeline: str
    fees: str
    eligibility: str
    process: str
    contact: str
    scenarios: tuple[ScenarioRule, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MatchedRule:
    rule: TopicRule
    score: int
    specificity_bonus: int
    scenario_hits: int
    term_hits: int
    context_bonus: int

    @property
    def explicit_hits(self) -> int:
        return self.term_hits + self.scenario_hits


_TOPIC_RULES: tuple[TopicRule, ...] = (
    TopicRule(
        topic="refund_exchange",
        terms=("退货", "换货", "退换货", "退款"),
        overview="退换货和退款通常要结合订单状态、签收时间、商品是否使用以及平台规则综合判断。",
        materials="建议先准备订单号、购买渠道、签收时间、商品当前状态，以及商品照片或开箱视频。",
        timeline="如果符合条件，退款到账时间通常还要看支付方式和平台审核进度，实际时效以原支付渠道处理结果为准。",
        fees="是否需要承担运费，通常取决于退换原因、商品是否存在质量问题以及平台售后规则。",
        eligibility="若商品未使用、仍在退换时效内，通常更容易申请退货、换货或退款；是否最终通过仍要以售后审核为准。",
        process="一般建议先在订单页发起售后申请，并同步说明问题原因、商品状态和诉求，必要时上传图片或视频证据。",
        contact="如果页面无法直接发起售后，建议携带订单号和证据材料联系人工客服进一步处理。",
        scenarios=(
            ScenarioRule(
                name="size_exchange",
                terms=("更大", "更小", "大一号", "小一号", "换尺寸", "换尺码", "尺码", "尺寸差价", "补差价", "差价"),
                overview="如果是想把商品更换成更大或更小的尺寸，通常要先看是否仍在换货时效内、商品是否适合尺码更换，以及平台是否支持补差价处理。",
                materials="建议准备订单号、当前商品规格、想更换的目标尺寸、商品状态说明，以及外包装和配件是否完整的照片。",
                timeline="尺寸换货和差价处理时效通常取决于订单状态、换货审核、库存以及补差价流程。",
                fees="是否需要补差价、退差价或承担运费，通常要看新旧规格价格、换货原因以及平台规则；无质量问题时运费更可能由买家承担。",
                eligibility="是否支持更换尺寸，通常取决于商品是否未明显使用、是否仍在换货时效内，以及目标尺寸是否有库存。",
                process="建议先确认目标尺寸是否有库存，再在订单页发起换货申请并说明需要更换的规格；如涉及差价，按页面或客服指引补差或退差。",
                contact="如果页面没有尺寸换货入口或差价规则写得不清楚，建议带上订单号联系人工客服核实。",
            ),
            ScenarioRule(
                name="refund_arrival",
                terms=("多久到账", "多久能到账", "原路返回", "信用卡", "银行卡", "储蓄卡", "借记卡", "退款多久"),
                overview="如果你主要关心退款到账，通常会先看支付渠道和退款是否已经审核通过，再判断到账时间。",
                materials="建议准备订单号、退款申请记录、支付方式、扣款账单或银行卡/信用卡账单截图。",
                timeline="退款到账时间通常取决于平台审核进度和支付渠道处理速度；信用卡、银行卡或第三方支付到账时效会不同。",
                fees="退款到账本身通常不以额外费用为核心，但是否有手续费或汇率差异，要看支付渠道规则。",
                eligibility="如果订单已成功取消或售后审核通过，通常才会进入正式退款流程。",
                process="建议先确认订单是否已经取消成功或售后是否审核通过，再查看退款状态和原支付渠道回执。",
                contact="如果长时间未到账，建议带上订单号和账单截图联系人工客服或支付渠道客服核查。",
            ),
            ScenarioRule(
                name="seven_day_no_reason",
                terms=("7天无理由", "七天无理由", "不想要了", "不喜欢", "不合适", "买错了"),
                overview="如果是未使用、配件齐全且仍在 7 天无理由范围内，通常更适合按无理由退货处理。",
                materials="通常需要订单号、签收时间，以及商品未使用、外包装和配件完整的说明或照片。",
                timeline="无理由退货的退款时效通常取决于退货寄回、仓库签收和平台审核进度。",
                fees="无质量问题的无理由退货，运费通常更可能由买家承担；是否支持运费险抵扣要看平台规则。",
                eligibility="是否能走 7 天无理由，通常取决于商品类目、是否拆封使用以及是否仍在时效内。",
                process="建议先在订单页选择无理由退货，并说明商品未使用、配件齐全的状态。",
                contact="如果页面提示该商品不支持无理由退货，建议带上订单号联系人工客服核实类目限制。",
            ),
            ScenarioRule(
                name="quality_reason",
                terms=("质量问题", "有问题", "故障", "坏了", "破损", "瑕疵"),
                overview="如果退货诉求是因为质量问题，更适合按质量售后或故障退换处理，而不是普通无理由退货。",
                materials="建议准备故障照片、视频、异常描述以及签收时间，最好保留开箱证据。",
                timeline="质量问题的审核和退款时效通常会受售后判责、检测流程和退款路径影响。",
                fees="若经核实属于质量问题，运费和退换责任通常更可能由商家或平台承担。",
                eligibility="是否支持直接退款或换货，通常取决于问题严重程度、签收时间和证据完整度。",
                process="建议先提交质量问题售后申请，并尽量上传清晰的故障照片、视频或开箱记录。",
                contact="如果页面无法上传证据或判责有争议，建议尽快联系人工客服升级处理。",
            ),
            ScenarioRule(
                name="opened_or_used",
                terms=("拆封", "开封", "使用过", "激活", "安装过", "试用过"),
                overview="如果商品已经拆封、激活或安装使用，通常更需要结合平台规则和售后审核判断是否还能退换。",
                materials="建议准备订单号、当前商品状态、拆封或使用情况说明，以及是否还能恢复完整包装的照片。",
                timeline="已拆封商品的审核时效通常取决于卖家判定和是否需要进一步检测。",
                fees="已拆封或已使用商品是否承担运费，通常更依赖具体类目规则和售后审核结果。",
                eligibility="是否还能退换，通常要看是否影响二次销售、是否存在质量问题以及平台规则。",
                process="建议先如实说明拆封和使用情况，再提交售后申请，避免因信息不一致影响审核。",
                contact="如果你不确定当前状态是否还能退换，建议带上订单号联系人工客服先做规则确认。",
            ),
            ScenarioRule(
                name="refund_rejected",
                terms=("驳回", "拒绝退款", "不通过", "审核失败", "没通过"),
                overview="如果退款或退货申请已经被驳回，通常需要先看驳回原因，再决定是补证据、改诉求还是升级申诉。",
                materials="建议准备订单号、售后单截图、驳回原因说明，以及商品当前状态、聊天记录或问题证据。",
                timeline="被驳回后的再次审核时效通常取决于补充材料是否完整以及平台工单处理进度。",
                fees="是否仍涉及运费或退款扣减，通常要结合驳回原因、商品状态和平台规则确认。",
                eligibility="是否还能重新提交，通常取决于驳回原因、是否仍在售后时效内以及是否有新的证据补充。",
                process="建议先查看驳回原因；如果是材料不足，就补齐证据重新提交；如果你认为判定不合理，可发起申诉或联系客服升级。",
                contact="如果页面没有再次提交入口，建议带上售后单截图联系人工客服复核。",
            ),
        ),
    ),
    TopicRule(
        topic="invoice",
        terms=("发票", "开发票", "抬头", "补开", "重开"),
        overview="发票问题通常需要确认订单号、开票类型、抬头信息以及当前开票状态。",
        materials="建议先准备订单号、开票类型、发票抬头、税号和接收邮箱等信息。",
        timeline="开票和补开发票的处理时效通常与订单状态、开票系统和平台规则有关，实际时间以平台处理进度为准。",
        fees="发票本身是否收费通常要看平台政策；如果涉及重开或邮寄，是否额外收费也要以平台规则为准。",
        eligibility="是否支持重开或修改抬头，通常取决于发票是否已开具以及平台是否允许更改。",
        process="一般建议先确认订单是否满足开票条件，再提交或修改开票信息，必要时联系人工客服协助处理。",
        contact="如果你不确定发票状态，建议携带订单号直接联系人工客服核实。",
        scenarios=(
            ScenarioRule(
                name="invoice_type",
                terms=("发票类型", "专票", "普票", "电子发票", "纸质发票"),
                overview="如果你想确认能开什么类型的发票，通常要先看订单开票入口支持的发票种类和商品类目限制。",
                materials="建议准备订单号、开票抬头、税号、邮箱，以及当前页面的开票选项截图。",
                timeline="发票开具时效通常取决于订单状态和开票系统处理进度。",
                fees="普通开票通常不以费用为核心；若涉及纸质邮寄或重开，是否收费要看平台规则。",
                eligibility="是否支持电子发票、普通发票或专用发票，通常要以订单开票入口和平台规则为准。",
                process="建议先进入订单页查看可选发票类型；如果没有目标类型，再联系人工客服核实。",
                contact="如果你需要确认企业专票或纸质发票，建议带上订单号和开票需求联系人工客服。",
            ),
            ScenarioRule(
                name="invoice_reissue",
                terms=("重开", "重开发票", "开错", "抬头填错", "税号错", "邮箱错"),
                overview="如果发票信息填错或已经开错，通常要先确认发票是否已开具、平台是否支持重开以及错误项属于哪一类。",
                materials="建议准备订单号、错误发票截图、正确的抬头/税号/邮箱信息，以及原申请记录。",
                timeline="发票重开的处理时效通常要看原发票状态、财务流程和平台规则，实际进度以系统审核为准。",
                fees="是否会产生邮寄费或重开成本，通常要结合平台规则和当前发票形态确认。",
                eligibility="是否支持重开，通常取决于发票是否已经开具、是否已报销以及平台是否允许修改该字段。",
                process="建议先确认错误项属于抬头、税号还是邮箱；若支持修改或重开，尽快在订单页或客服渠道提交更正申请。",
                contact="如果你不确定当前发票是否还能重开，建议带上订单号和错误发票截图联系人工客服核实。",
            ),
            ScenarioRule(
                name="invoice_after_issued",
                terms=("已经开票", "已开票", "开出来了", "开完了", "发票已出"),
                overview="如果发票已经开具，后续想修改抬头、税号或邮箱，通常不能按普通开票重新点选，而是要先看平台是否支持更正、红冲或重开。",
                materials="建议准备订单号、已开具发票截图、正确的抬头/税号/邮箱信息和当前修改诉求。",
                timeline="已开具发票的修改或重开时效通常取决于财务审核和平台发票流程。",
                fees="是否产生额外费用通常要看平台是否支持免费更正或重开，以及是否涉及邮寄。",
                eligibility="是否还能改，通常取决于发票是否已报销、是否已入账以及平台规则是否允许。",
                process="建议先确认发票状态，再申请更正、红冲或重开；如果页面没有入口，直接联系人工客服核实。",
                contact="如果你已经收到开票完成通知，建议一并提供发票截图给人工客服，方便快速判断还能怎么改。",
            ),
        ),
    ),
    TopicRule(
        topic="shipping",
        terms=("物流", "快递", "发货", "补发", "签收", "运费", "乡镇", "国外", "配送", "丢件", "退回", "异常"),
        overview="物流和配送问题通常需要结合订单号、物流单号、收货地址以及当前物流状态判断。",
        materials="建议先准备订单号、物流单号、收货地址、异常时间点以及相关截图。",
        timeline="物流时效通常受仓库出库、承运商揽收和目的地配送能力影响，实际到达时间以物流轨迹为准。",
        fees="是否产生额外运费，通常要看配送地区、补发原因和物流方案，需结合订单系统进一步确认。",
        eligibility="乡镇、海外或特殊地区配送是否支持，通常要看平台覆盖范围和商品限制。",
        process="建议先核对物流轨迹、签收状态和地址信息；若长期停滞、少件或误签，可尽快发起物流异常反馈。",
        contact="如物流长时间无更新或疑似丢件，建议携带订单号和物流单号联系人工客服或承运商处理。",
        scenarios=(
            ScenarioRule(
                name="village_or_overseas",
                terms=("乡镇", "农村", "村里", "国外", "海外", "寄到国外", "国际", "偏远地区"),
                overview="如果你想确认乡镇、偏远地区或海外是否可配送，通常要先看商品是否支持该地区发货，以及末端配送能力是否覆盖。",
                materials="建议准备订单号、商品链接、详细收货地址或邮编，以及页面配送说明截图。",
                timeline="乡镇、偏远地区或海外配送时效通常会比普通城市地址更长，实际到达时间取决于仓库、承运商和目的地配送能力。",
                fees="是否需要额外运费，通常要看商品类目、配送地区和物流方案；偏远地区或海外更可能产生附加费用。",
                eligibility="是否支持送达，通常取决于商品限制、地区覆盖和平台物流规则，不是所有商品都支持乡镇或海外直送。",
                process="建议先在商品页或下单页核对配送范围；如果系统未明确显示，再联系人工客服确认详细地址是否可送达。",
                contact="如果你已经有完整地址或邮编，建议一并提供给人工客服，这样通常更容易直接核实。",
            ),
            ScenarioRule(
                name="tracking_stalled",
                terms=("没更新", "不更新", "没动", "停滞", "卡住", "未揽收", "一直没有物流"),
                overview="如果物流长时间没有更新，更像是揽收延迟、中转停滞或系统轨迹未同步的场景。",
                materials="建议准备订单号、物流单号、最近一次轨迹时间以及页面截图。",
                timeline="这类物流停滞的处理时效通常要看仓库、承运商反馈以及是否需要人工催查。",
                fees="物流停滞本身通常不是收费问题，重点是先确认是否超过承诺时效。",
                eligibility="是否能按异常物流处理，通常要对照最近轨迹时间和平台承诺时效判断。",
                process="建议先核对最近一次轨迹更新时间；如果明显超时，可提交物流催查或人工催件。",
                contact="如果轨迹超过较长时间仍无变化，建议同时联系人工客服和承运商催查。",
            ),
            ScenarioRule(
                name="package_lost_or_returned",
                terms=("丢件", "找不到", "退回", "被退回", "拒收", "错发", "异常轨迹", "运回"),
                overview="如果包裹出现丢件、退回、拒收或异常回流，通常要先按物流异常或包裹回查场景处理，而不是直接默认已完成配送。",
                materials="建议准备订单号、物流单号、最近轨迹截图、签收状态和收件地址信息。",
                timeline="丢件、退回或异常回流的核查时效通常要看承运商回查和仓库处理速度。",
                fees="这类问题是否涉及补发运费或退回费用，通常要看责任归属和平台规则。",
                eligibility="是否能按丢件或异常退回处理，通常要看轨迹是否完整、是否已签收以及承运商核查结果。",
                process="建议先联系承运商确认包裹当前状态，再在平台提交物流异常或退回异常申请。",
                contact="如果物流显示退回但你并未发起退货，建议尽快联系人工客服和承运商双向核实。",
            ),
            ScenarioRule(
                name="signed_but_missing",
                terms=("已签收", "显示签收", "代签", "没收到", "未收到"),
                overview="如果物流显示已签收但实际未收到，更像是代签、误投或末端配送异常场景。",
                materials="建议准备订单号、物流单号、签收时间、签收截图，以及门卫、驿站或家人是否代收的信息。",
                timeline="已签收未收到通常需要先核实末端签收情况，处理时效取决于承运商回查速度。",
                fees="这类问题通常不以费用为主，重点是先确认签收责任和包裹去向。",
                eligibility="是否可以按丢件或误签处理，通常要看签收记录、代签情况和回查结果。",
                process="建议先联系承运商核实签收人，再同步在平台提交物流异常或未收到申请。",
                contact="如果承运商和平台说法不一致，建议带上轨迹截图联系人工客服升级处理。",
            ),
            ScenarioRule(
                name="redirect_or_wrong_address",
                terms=("改派", "拦截", "送错", "发错地址", "派错", "改地址"),
                overview="如果是改派、拦截或送错地址，通常要先确认订单地址、物流状态以及末端站点是否支持改派。",
                materials="建议准备订单号、物流单号、原地址、新地址和当前轨迹截图。",
                timeline="地址改派或拦截的处理时效通常取决于包裹是否已进入末端配送以及承运商是否支持操作。",
                fees="是否收费通常要看改派距离、承运商政策和是否需要二次派送。",
                eligibility="是否还能改派，通常取决于包裹是否已签收、是否到站以及承运商规则。",
                process="建议先联系承运商确认是否支持改派，再在平台侧同步更新或备注地址异常。",
                contact="如果系统侧和物流侧信息不一致，建议携带订单号联系人工客服协调处理。",
            ),
        ),
    ),
    TopicRule(
        topic="complaint",
        terms=("投诉", "虚假宣传", "假货", "二手", "辱骂", "赔偿", "保质期", "临近过期"),
        overview="投诉、假货、虚假宣传、二手商品或服务态度问题通常需要先固定证据再升级处理。",
        materials="建议准备订单记录、聊天记录、商品照片、视频以及页面宣传截图等证据。",
        timeline="投诉和升级处理的时效通常取决于平台工单流程和证据完整度，实际进度以平台反馈为准。",
        fees="投诉本身是否涉及额外费用通常不是重点，更重要的是先固定证据并确认赔付依据。",
        eligibility="是否支持赔偿或升级处理，通常要看证据是否充分以及平台判责结果。",
        process="建议优先通过平台投诉单、售后工单或人工客服渠道提交证据并说明诉求。",
        contact="如果问题较严重，建议直接联系人工客服并走官方投诉流程。",
    ),
    TopicRule(
        topic="after_sales",
        terms=("售后", "维修", "保修", "保修卡", "保障卡", "维修费用"),
        overview="售后、维修和保修问题通常需要确认商品型号、故障现象、购买时间和保修凭证。",
        materials="建议准备订单号、商品型号、故障描述、故障照片或视频以及保修凭证。",
        timeline="维修和保修审核时效通常要看检测流程、备件情况和售后网点安排，实际时间以售后反馈为准。",
        fees="是否收费通常取决于是否人为损坏、是否在保修期内以及具体维修项目。",
        eligibility="若商品仍在保修期且非人为损坏，通常更容易进入保修流程；最终结果仍以售后检测为准。",
        process="建议先提交售后申请并说明故障现象，必要时配合寄修、检测或预约售后网点。",
        contact="如果你不确定是否在保修范围内，建议携带凭证联系人工客服或官方售后确认。",
        scenarios=(
            ScenarioRule(
                name="in_warranty",
                terms=("在保", "保修期内", "没过保", "还在保修期"),
                overview="如果确认仍在保修期内，通常更适合先按保修检测流程推进。",
                materials="建议准备订单号、购买时间、保修凭证、型号信息和故障照片或视频。",
                timeline="在保维修的处理时效通常取决于检测排队、配件库存和网点安排。",
                fees="如果检测后确认为非人为损坏且仍在保修范围内，通常更可能免维修费；运费是否减免仍要看政策。",
                eligibility="是否最终按保修处理，还要看故障原因、购买时间和检测结论是否一致。",
                process="建议先提交保修申请，并在描述里明确购买时间、故障现象和是否仍在保修期内。",
                contact="如果你已经有保修凭证，建议直接联系官方售后或人工客服加快核验。",
            ),
            ScenarioRule(
                name="out_of_warranty",
                terms=("过保", "超出保修期", "保修过了", "过了保修期"),
                overview="如果已经过保，通常更可能进入收费维修或付费检测流程，而不是标准保修。",
                materials="建议准备订单号、商品型号、购买时间和当前故障情况说明。",
                timeline="过保维修的处理时效通常取决于检测、报价确认和备件供应情况。",
                fees="过保后通常更可能产生检测费、维修费或配件费，具体仍要以售后报价为准。",
                eligibility="是否还能获得免费处理，通常只有在特殊活动、延保或明确质量责任场景下才更有可能。",
                process="建议先咨询售后是否支持付费检测，再确认报价和是否值得维修。",
                contact="如果你拿不准是否真的过保，建议带上订单号和购买时间联系人工客服先核验。",
            ),
            ScenarioRule(
                name="human_damage",
                terms=("人为", "进水", "摔坏", "磕碰", "私拆", "外力"),
                overview="如果故障涉及进水、摔落、磕碰或私拆，更像是人为损坏场景，通常会影响是否能走保修。",
                materials="建议准备故障照片、受损部位说明、购买时间和当前是否还能开机的情况。",
                timeline="人为损坏的处理时效通常取决于检测、责任认定和维修报价确认。",
                fees="如果最终判定为人为损坏，通常更可能产生检测费或维修费。",
                eligibility="这类情况是否还能免费保修，通常要以售后检测和品牌规则为准，很多场景下更可能进入付费维修。",
                process="建议先提交检测或维修申请，并如实说明进水、跌落或私拆等情况，避免后续判责反复。",
                contact="如果你担心被误判，建议在寄修前带上照片和订单号联系人工客服或官方售后做预判。",
            ),
            ScenarioRule(
                name="repair_delay_or_repeat_failure",
                terms=("半个月", "一直没修好", "还没修好", "返修后", "又坏了", "同样的故障", "维修不彻底", "久修未好"),
                overview="如果维修时间明显过长，或维修后短时间内再次出现同样故障，通常要先按维修进度异常或返修复发场景升级核查。",
                materials="建议准备订单号、维修单号、送修时间、故障复发说明，以及维修前后照片、视频或聊天记录。",
                timeline="维修延期或返修复发的处理时效通常取决于售后网点反馈、复检安排和备件情况；如果需要重新返修，周期可能重新计算。",
                fees="如果确认是上次维修不彻底或返修复发，通常更适合优先核实是否应免费复检、免费返修或延长维修保障。",
                eligibility="是否支持优先处理、免费复检或延长保障，通常取决于维修记录、故障一致性和售后复核结果。",
                process="建议先提供维修单号和送修时间，要求售后核实当前维修进度；如果是返修后短时间内同故障复发，可明确说明希望走复检或重新返修流程。",
                contact="如果网点反馈不清晰或长时间没有结果，建议带上维修单截图联系人工客服或官方售后升级协调。",
            ),
            ScenarioRule(
                name="rejected_or_disputed",
                terms=("驳回", "拒绝", "不通过", "判定不合理", "想申诉", "复核", "重新审核", "证据不足"),
                overview="如果售后申请被驳回或你认为判定不合理，通常要先看驳回原因，再决定是补证据、申诉还是升级人工复核。",
                materials="建议准备订单号、售后单号、驳回截图、故障证据、聊天记录和你希望达到的处理诉求。",
                timeline="驳回后的再次审核或复核时效通常取决于补充材料是否完整以及平台工单处理速度。",
                fees="复核本身通常不涉及额外费用，但后续是否产生检测费、维修费或运费，要看最终判责结果。",
                eligibility="是否还能重新申诉，通常取决于是否仍在售后时效内，以及是否能补充更充分的证据。",
                process="建议先明确驳回原因；如果是证据不足，就补充照片、视频或聊天记录后重新提交；如果判定有争议，可直接申请人工复核。",
                contact="如果平台没有明显的复核入口，建议带上售后截图联系人工客服说明你希望重新审核的理由。",
            ),
        ),
    ),
    TopicRule(
        topic="quality_issue",
        terms=("划痕", "破损", "瑕疵", "少件", "少了一件", "保质期", "过期", "临期"),
        overview="破损、瑕疵、少件或保质期异常这类问题，通常需要尽快留证并核对签收时间。",
        materials="建议准备订单号、签收时间、问题照片、开箱视频以及异常细节说明。",
        timeline="若问题发生在签收后较短时间内，通常更容易进入补发、换货或售后处理流程。",
        fees="是否产生额外费用通常要看问题原因和平台售后规则，质量问题一般更适合优先申请售后判责。",
        eligibility="是否支持补发、换货或退款，通常要看问题是否属于运输或质量异常，以及证据是否充分。",
        process="建议先在订单页发起售后，并同步上传图片或视频说明问题。",
        contact="如系统无法提交售后，建议直接联系人工客服升级处理。",
        scenarios=(
            ScenarioRule(
                name="packaging_damage",
                terms=("包装破损", "外包装破损", "包装损坏", "封条破损", "箱子破了"),
                overview="如果主要是外包装破损，先要区分是仅包装受损，还是已经影响到商品本体、配件或签收状态。",
                materials="建议准备订单号、签收时间、外包装照片、商品本体照片以及开箱视频。",
                timeline="包装破损类售后通常越早报备越容易核实，处理时效取决于售后审核和物流判责。",
                fees="如果最终判定为运输或质量问题，额外运费通常不应由买家单独承担；具体仍要看售后审核。",
                eligibility="仅外包装异常不一定直接影响退换货，但是否支持换货、退款或补偿，要看商品是否受损和证据是否充分。",
                process="建议先拍照留证，再核对商品本体和配件是否完好；如有异常，尽快发起破损售后并上传证据。",
                contact="如果你担心影响退换，建议带上签收时间和开箱证据联系人工客服先做判责。",
            ),
            ScenarioRule(
                name="missing_items",
                terms=("少发", "少件", "漏发", "缺件", "少了一件"),
                overview="如果收到后发现少发、漏发或缺件，通常可以先按缺件或补发场景申请售后核查。",
                materials="建议准备订单号、包装清单、缺件照片、开箱视频和签收时间。",
                timeline="缺件或少发的处理时效通常取决于售后审核、仓库核查和补发物流安排。",
                fees="若核实属于原包装缺失或漏发，补发费用通常不应由买家承担。",
                eligibility="是否支持补发、换货或退款，通常要看缺少的是主商品还是配件，以及证据是否充分。",
                process="建议先对照包装清单确认缺少内容，再发起缺件/补发申请，并明确说明缺少的是哪一项。",
                contact="如果系统无法选择缺件类型，建议带上包装清单和开箱证据联系人工客服处理。",
            ),
        ),
    ),
    TopicRule(
        topic="order_change",
        terms=("取消订单", "订单", "到账", "信用卡", "原路返回"),
        overview="订单取消和退款到账问题通常取决于订单是否发货、支付方式以及平台规则。",
        materials="建议准备订单号、支付方式、支付时间和相关账单截图。",
        timeline="退款一般按原支付路径退回，到账时间通常受支付渠道和平台审核流程影响。",
        fees="是否产生手续费通常要看支付渠道和订单场景，具体仍要以支付渠道规则为准。",
        eligibility="如果订单尚未发货，通常更容易申请取消；已发货订单则可能需要走拒收或售后流程。",
        process="建议先确认订单状态，再在订单页提交取消或售后申请。",
        contact="若页面状态异常，建议携带订单号联系人工客服核实。",
        scenarios=(
            ScenarioRule(
                name="refund_arrival",
                terms=("多久到账", "多久能到账", "原路返回", "信用卡", "退款多久"),
                overview="如果你主要关心退款到账，通常会按原支付路径退回；是否已经发起退款、以及支付渠道处理速度会直接影响到账时间。",
                materials="建议准备订单号、退款申请记录、支付方式、扣款账单或信用卡账单截图。",
                timeline="退款到账时间通常取决于平台审核进度和支付渠道处理速度；信用卡类支付一般也按原路径退回。",
                fees="退款到账本身通常不以额外费用为核心，但是否有手续费或汇率差异，要看支付渠道规则。",
                eligibility="如果订单已成功取消或售后审核通过，通常才会进入正式退款流程。",
                process="建议先确认订单是否已经取消成功或售后是否审核通过，再查看退款状态和原支付渠道回执。",
                contact="如果长时间未到账，建议带上订单号和账单截图联系人工客服或支付渠道客服核查。",
            ),
        ),
    ),
    TopicRule(
        topic="manual_request",
        terms=("纸质版说明书", "电子版说明书", "电子版", "说明书"),
        overview="说明书补发或获取电子版通常需要先确认具体商品名称和型号。",
        materials="建议准备商品名称、型号、订单号以及需要的说明书版本。",
        timeline="电子版通常更容易即时获取；纸质版是否支持补寄以及时效，通常要看包装配置和售后政策。",
        fees="纸质版补寄是否收费通常要以商品售后规则和客服核实结果为准。",
        eligibility="是否支持补寄纸质说明书，通常要看商品原始包装清单和品牌政策。",
        process="建议优先通过商品页面、品牌官网或客服渠道获取电子版，再确认是否支持纸质补寄。",
        contact="如果页面找不到说明书入口，建议携带型号联系人工客服。",
    ),
    TopicRule(
        topic="platform_service",
        terms=("以旧换新", "优惠券", "试用装", "试用", "智能客服", "人工客服", "活动规则", "会员", "优惠", "价格咨询"),
        overview="平台活动、优惠、试用和智能客服能力通常要以商品页面、活动规则和平台实际入口为准。",
        materials="建议准备商品名称、订单号、活动页面截图、优惠券信息或客服对话截图。",
        timeline="活动资格、优惠券使用和人工客服处理时效通常取决于平台规则和当前排队情况。",
        fees="是否产生费用或能否抵扣，通常要看活动规则、优惠券门槛和商品适用范围。",
        eligibility="是否支持以旧换新、试用装、优惠券或会员权益，通常取决于商品类目、活动时间和账号资格。",
        process="建议先查看商品页、活动页或结算页是否有对应入口；如果没有入口，再联系人工客服核实。",
        contact="智能客服可处理常见订单、物流、售后和发票问题；需要人工核实时，建议转人工并提供订单号和截图。",
        scenarios=(
            ScenarioRule(
                name="trade_in",
                terms=("以旧换新", "旧机", "回收"),
                overview="以旧换新通常取决于商品类目、旧机状态、活动入口和平台估价规则。",
                materials="建议准备新商品链接、旧机型号、成色说明、估价截图和订单信息。",
                timeline="以旧换新的处理时效通常取决于旧机评估、回收验机和补贴发放流程。",
                fees="是否有补贴、回收价或额外费用，要以活动页面和旧机检测结果为准。",
                eligibility="是否符合以旧换新资格，通常要看旧机品类、活动时间和账号/地区限制。",
                process="建议先在商品页查看是否有以旧换新入口，再按页面要求填写旧机信息并确认估价。",
                contact="如果页面没有入口或估价异常，建议联系人工客服核实活动资格。",
            ),
            ScenarioRule(
                name="coupon",
                terms=("优惠券", "优惠", "满减", "券", "折扣"),
                overview="优惠券能否使用通常取决于有效期、使用门槛、适用品类和是否可叠加。",
                materials="建议准备优惠券截图、商品链接、结算页截图和账号信息。",
                timeline="优惠券通常需要在有效期内使用，过期或活动结束后一般较难补用。",
                fees="优惠券本身不一定收费，重点是是否满足门槛以及是否能与其他活动叠加。",
                eligibility="是否适用所有商品，要以优惠券详情中的适用范围和排除规则为准。",
                process="建议在结算页查看是否可勾选优惠券；若不可用，先核对有效期、门槛和适用品类。",
                contact="如果优惠券满足规则但无法使用，建议带截图联系人工客服核实。",
            ),
            ScenarioRule(
                name="trial_sample",
                terms=("试用装", "试用", "样品"),
                overview="是否提供试用装或试用服务通常取决于商品活动、库存和平台规则。",
                materials="建议准备商品名称、活动页面截图和账号信息。",
                timeline="试用申请和发放时效通常以活动页面说明为准。",
                fees="试用是否收费、是否需要押金或运费，要以活动规则为准。",
                eligibility="是否能申请试用，通常取决于活动名额、账号资格和商品类目。",
                process="建议先查看商品页或活动页是否有试用入口，再按页面要求提交申请。",
                contact="如果页面没有试用入口，建议联系人工客服确认是否有相关活动。",
            ),
            ScenarioRule(
                name="trial_extension_or_fault",
                terms=("试用期", "试用期间", "延长试用", "延长试用期限", "试用故障", "试用期间故障"),
                overview="试用期间出现故障、想延长试用期限或确认能否换货时，通常要同时看活动规则、试用协议和故障责任判定。",
                materials="建议准备订单号、商品名称、活动页面或试用协议截图、故障照片/视频，以及试用开始时间和当前状态说明。",
                timeline="试用延期、故障审核或换货处理时效通常取决于活动规则、客服审核和售后判责进度。",
                fees="是否需要承担运费、押金或其他费用，通常要看试用活动规则、故障责任和换货方式。",
                eligibility="是否能延长试用或直接更换故障商品，通常取决于活动是否允许延期、商品是否仍在试用期内，以及故障是否属于非人为原因。",
                process="建议先核对试用活动规则，再提交故障说明和延期/换货诉求；如果活动页没有自助入口，尽快联系人工客服登记并申请处理。",
                contact="如果同时涉及试用延期和故障换货，建议一次性提供订单号、活动截图和故障证据给人工客服，避免重复沟通。",
            ),
            ScenarioRule(
                name="smart_customer_service",
                terms=("智能客服", "人工客服", "解答不了", "转人工"),
                overview="智能客服通常适合处理订单、物流、退换货、发票和售后等常见问题。",
                materials="如果要转人工，建议准备订单号、商品名称、问题截图和已尝试过的处理步骤。",
                timeline="人工客服响应时效通常取决于当前排队情况和问题复杂度。",
                fees="联系客服本身通常不是收费问题，重点是提供充分信息以减少来回沟通。",
                eligibility="需要订单核实、复杂判责或投诉升级的问题，通常更适合转人工处理。",
                process="建议先让智能客服识别问题类型；如果无法解决，再选择转人工并一次性提供关键信息。",
                contact="转人工后建议直接说明诉求、订单号、异常现象和期望处理方式。",
            ),
        ),
    ),
    TopicRule(
        topic="address_change",
        terms=("改地址", "修改地址", "收货地址", "改收件地址", "改配送地址"),
        overview="修改收货地址通常要先看订单是否已经发货或进入出库流程。",
        materials="建议准备订单号、当前订单状态和要修改的新地址。",
        timeline="如果订单还未出库，通常更容易修改；若已发货，时效一般要看物流拦截或改派结果。",
        fees="是否产生额外费用通常要看改派方式、配送地区和平台规则。",
        eligibility="是否还能修改地址，通常取决于订单是否已出库、已发货以及物流是否支持改派。",
        process="建议先确认订单状态；若仍可修改，尽快在订单页或客服渠道提交地址变更申请。",
        contact="若系统内无法直接改地址，建议立即联系人工客服或物流方协助处理。",
        scenarios=(
            ScenarioRule(
                name="address_after_shipment",
                terms=("已经发货", "已发货", "出库了", "配送中", "快递已经发出"),
                overview="如果订单已经发货或进入配送中，修改地址通常不再是普通订单修改，而是要看物流是否支持拦截、改派或末端改址。",
                materials="建议准备订单号、物流单号、原地址、新地址和当前物流轨迹截图。",
                timeline="已发货后改地址的处理时效通常取决于包裹当前节点和承运商是否支持改派，越早处理越容易成功。",
                fees="已发货后改派是否收费，通常要看承运商规则、改派距离和是否需要二次派送。",
                eligibility="是否还能改地址，通常取决于是否已签收、是否进入末端配送，以及承运商是否允许改派。",
                process="建议先联系承运商确认是否支持改派，同时在平台侧提交地址变更说明，避免订单侧和物流侧信息不一致。",
                contact="如果你这边已经有物流单号，建议一并提供给人工客服或物流客服，这样通常更快确认是否能改派。",
            ),
        ),
    ),
    TopicRule(
        topic="installation_service",
        terms=("预约安装", "上门安装", "安装服务", "师傅安装", "安装预约"),
        overview="预约安装或上门安装服务通常要结合订单状态、商品型号、安装地址和服务覆盖范围判断。",
        materials="建议准备订单号、商品型号、安装地址、联系电话和可预约时间。",
        timeline="预约安装时效通常受地区覆盖和师傅排期影响，最早可预约时间要以系统排期为准。",
        fees="是否收费通常要看商品是否含安装服务、所在地区以及安装项目范围。",
        eligibility="是否支持上门安装，通常要看商品类目、服务覆盖地区和订单状态。",
        process="建议先确认订单是否支持安装服务，再提交预约申请并填写地址与时间。",
        contact="如系统内没有安装入口，建议带上订单号联系人工客服确认。",
        scenarios=(
            ScenarioRule(
                name="installation_reschedule",
                terms=("改约", "改时间", "重新预约", "约不上", "师傅没来", "爽约", "改安装时间"),
                overview="如果已经预约安装但需要改约，或师傅未按约到场，通常要先确认原预约状态和最近可改约档期。",
                materials="建议准备订单号、预约时间、安装地址、联系电话，以及系统预约截图或师傅联系记录。",
                timeline="改约或重新排期的时效通常取决于地区覆盖、师傅排班和最近可预约时间段。",
                fees="是否产生额外费用，通常要看改约原因、是否多次爽约以及平台安装服务规则。",
                eligibility="是否支持改约，通常取决于服务单状态、地区排期和是否已经上门。",
                process="建议先在安装服务单里查看是否支持改约；若系统无法修改，可联系平台或安装师傅重新确认时间。",
                contact="如果师傅爽约或多次改期，建议带上服务单截图联系人工客服升级协调。",
            ),
        ),
    ),
    TopicRule(
        topic="price_protection",
        terms=("保价", "价保", "降价", "买贵了"),
        overview="价保或降价补差通常要结合订单时间、活动规则和平台价保政策判断。",
        materials="建议准备订单号、下单时间、当前商品页面价格截图以及活动页面信息。",
        timeline="是否在价保周期内通常是关键，具体审核时间仍要以平台处理进度为准。",
        fees="价保通常不是收费问题，关键在于是否满足平台补差规则。",
        eligibility="是否支持补差，通常取决于商品是否参加价保、下单时间是否在周期内以及活动是否适用。",
        process="建议先核对商品页面的价保规则，再提交价保申请或联系人工客服核实。",
        contact="如果你不确定是否在价保周期内，建议携带订单号和截图联系人工客服。",
    ),
    TopicRule(
        topic="payment_issue",
        terms=("支付失败", "付款失败", "扣款", "重复扣款", "支付异常", "扣了两次"),
        overview="支付异常问题通常需要先核对订单状态、支付流水和扣款时间。",
        materials="建议准备订单号、支付方式、扣款时间、银行或平台账单截图。",
        timeline="支付异常的处理时效通常取决于支付渠道回执、平台对账和财务核查进度。",
        fees="若涉及重复扣款或资金冻结，是否产生额外费用通常要以支付渠道处理结果为准。",
        eligibility="是否属于支付异常，通常要结合订单是否生成、是否重复扣款和账单记录综合判断。",
        process="建议先确认订单是否生成，再核对支付流水；若账单异常，尽快联系客服发起核查。",
        contact="如资金问题较紧急，建议同时联系平台客服和支付渠道客服处理。",
        scenarios=(
            ScenarioRule(
                name="repeat_charge",
                terms=("重复扣款", "扣了两次", "扣款两次", "多扣", "反复扣款", "重复支付"),
                overview="如果出现重复扣款，核心不是先重新下单，而是先核对是否生成了重复订单，以及两笔资金分别处于支付成功、冻结还是待退状态。",
                materials="建议准备订单号、支付流水号、两笔扣款账单截图、支付时间和支付方式信息。",
                timeline="重复扣款的核查时效通常取决于平台对账和支付渠道回执；如果一笔只是冻结款，解冻时间还要看银行或支付工具规则。",
                fees="是否会产生额外手续费通常要看支付渠道；多数情况下重点是确认哪一笔是有效支付、哪一笔需要退回或解冻。",
                eligibility="是否属于重复扣款，通常要结合订单数量、流水数量和账单状态综合判断，并不一定每次看到两笔记录都代表实际成功扣款两次。",
                process="建议先确认是否生成了重复订单，再核对每笔流水状态；如果只有一笔订单却有两笔成功扣款，应尽快联系客服发起支付异常核查。",
                contact="如果资金占用比较紧急，建议同时联系平台客服和支付渠道客服说明重复扣款情况。",
            ),
        ),
    ),
    TopicRule(
        topic="delivery_delay",
        terms=("催发货", "一直不发货", "迟迟不发货", "延迟发货", "发货慢"),
        overview="催发货和延迟发货问题通常要结合下单时间、库存状态和页面承诺时效判断。",
        materials="建议准备订单号、下单时间、商品页面承诺时效截图以及当前订单状态。",
        timeline="发货时间通常受库存、活动高峰和仓库排队影响，实际出库时间要以订单状态更新为准。",
        fees="是否涉及补偿或运费调整通常要以平台规则和实际延迟原因判断。",
        eligibility="是否属于异常延迟，通常要对照商品页面承诺时效和平台规则确认。",
        process="建议先核对订单状态和承诺时效；若明显超时，可发起催发货或联系人工客服跟进。",
        contact="如果订单长时间未出库，建议携带订单号联系人工客服催单。",
    ),
    TopicRule(
        topic="warranty_period",
        terms=("保修期", "质保期", "保修多久", "质保多久"),
        overview="保修期或质保期通常要结合商品类目、品牌规则、购买时间和保修凭证确认。",
        materials="建议准备商品名称、型号、下单时间和保修凭证。",
        timeline="保修期本身属于规则信息，是否还在有效期内通常要根据购买时间和官方规则核对。",
        fees="若超出保修期，通常更可能涉及收费维修；具体费用仍要以售后检测结果为准。",
        eligibility="是否在保修范围内，通常取决于购买时间、故障类型和是否有人为损坏。",
        process="建议先核对页面、保修卡或说明书中的保修规则，再确认当前是否还在期限内。",
        contact="如果你拿不准保修期起止时间，建议带上订单号联系人工客服或官方售后核实。",
    ),
    TopicRule(
        topic="accessory_request",
        terms=("配件", "附件", "包装盒", "缺少配件", "少配件", "补寄配件"),
        overview="配件、附件、包装盒或补寄问题通常要先确认商品包装清单和实际缺少的内容。",
        materials="建议准备订单号、商品名称、缺少的具体配件名称以及开箱照片或视频。",
        timeline="补寄处理时效通常取决于售后审核、配件库存和物流安排。",
        fees="是否收费通常要看缺件原因、是否属于原包装缺失以及平台售后规则。",
        eligibility="是否支持补寄，通常要结合包装清单、问题责任和售后规则判断。",
        process="建议先核对商品包装清单，再提交缺件或补寄申请，并说明缺少的具体部件。",
        contact="如页面无法提交补寄，建议带上订单号联系人工客服处理。",
        scenarios=(
            ScenarioRule(
                name="accessory_rejected",
                terms=("补寄失败", "补件失败", "驳回", "拒绝补寄", "不给补寄", "审核不通过"),
                overview="如果补寄配件申请被驳回，通常要先看驳回原因，是包装清单不符、超出售后时效，还是证据不足。",
                materials="建议准备订单号、包装清单截图、缺少配件照片、开箱视频和售后单驳回截图。",
                timeline="补件驳回后的复核时效通常要看补充材料是否充分以及客服工单进度。",
                fees="若最终不支持免费补寄，是否能付费补购通常要看品牌配件供应和平台规则。",
                eligibility="是否还能重新申请，通常取决于是否仍在售后时效内，以及是否能补充更完整的缺件证据。",
                process="建议先核对驳回原因；如果是证据不足，就补充开箱视频和包装清单后重新提交；如果是规则限制，可改为咨询付费补购。",
                contact="如果你认为包装内确实缺件但申请被拒，建议带上包装清单和售后截图联系人工客服复核。",
            ),
        ),
    ),
)

_DETAIL_INTENT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("materials", ("材料", "资料", "凭证", "截图", "证明", "准备什么", "需要什么", "提供什么")),
    ("timeline", ("多久", "几天", "多长时间", "什么时候", "何时", "多快", "时效")),
    ("fees", ("收费", "费用", "多少钱", "运费", "免费", "要钱", "花钱")),
    ("eligibility", ("可以吗", "能不能", "是否支持", "条件", "符合", "满足", "能否", "还能")),
    ("process", ("怎么办", "怎么处理", "流程", "怎么申请", "如何申请", "步骤", "怎么走")),
    ("contact", ("联系谁", "找谁", "人工客服", "客服电话", "找哪个部门")),
)

_FORCED_TOPIC_PRIORITY: tuple[str, ...] = (
    "refund_exchange",
    "shipping",
    "invoice",
    "quality_issue",
    "after_sales",
    "delivery_delay",
    "accessory_request",
    "complaint",
)


class CustomerServicePolicy:
    """Generate conservative but helpful replies for generic service scenarios."""

    def answer(self, question: str, *, context_topics: list[str] | None = None) -> PolicyResponse:
        normalized = question.strip()
        detail_intents = self._detect_detail_intents(normalized)
        matched_entries = self._match_rules(
            normalized,
            context_topics=context_topics,
            detail_intents=detail_intents,
        )

        if not matched_entries and context_topics:
            matched_entries = [
                MatchedRule(
                    rule=rule,
                    score=2,
                    specificity_bonus=0,
                    scenario_hits=0,
                    term_hits=0,
                    context_bonus=2,
                )
                for rule in _TOPIC_RULES
                if rule.topic in context_topics
            ]

        if not matched_entries:
            answer = (
                "您好，这类问题需要进一步确认订单和商品信息才能帮您准确处理。"
                "请提供订单号、商品名称和具体问题描述，我们会尽快为您安排合适的售后方案。"
            )
            return PolicyResponse(answer=answer, confidence=0.58, matched_topics=[])

        selected_entries = self._select_rules(
            matched_entries,
            question=normalized,
            detail_intents=detail_intents,
        )
        selected_rules = [entry.rule for entry in selected_entries]
        matched_scenarios = [self._match_scenario(rule, normalized) for rule in selected_rules]
        sections = [
            self._compose_topic_sections(
                rule,
                question=normalized,
                detail_intents=detail_intents,
                scenario=scenario,
            )
            for rule, scenario in zip(selected_rules, matched_scenarios)
        ]
        answer = self._compose_answer(
            question=normalized,
            sections=sections,
            matched_topics=[rule.topic for rule in selected_rules],
            detail_intents=detail_intents,
        )
        scenario_bonus = sum(1 for scenario in matched_scenarios if scenario is not None)
        confidence = min(
            0.7 + 0.03 * len({rule.topic for rule in selected_rules}) + 0.02 * len(detail_intents) + 0.01 * scenario_bonus,
            0.92,
        )
        return PolicyResponse(
            answer=answer,
            confidence=confidence,
            matched_topics=[rule.topic for rule in selected_rules],
        )

    def _match_rules(
        self,
        question: str,
        *,
        context_topics: list[str] | None = None,
        detail_intents: list[str] | None = None,
    ) -> list[MatchedRule]:
        scored: list[MatchedRule] = []
        detail_intents = detail_intents or []
        for rule in _TOPIC_RULES:
            matched_terms = [term for term in rule.terms if term in question]
            term_hits = len(matched_terms)
            scenario = self._match_scenario(rule, question)
            scenario_hits = 0
            specificity_bonus = max((len(term) for term in matched_terms), default=0)
            if scenario is not None:
                matched_scenario_terms = [term for term in scenario.terms if term in question]
                scenario_hits = len(matched_scenario_terms)
                specificity_bonus = max(
                    specificity_bonus,
                    max((len(term) for term in matched_scenario_terms), default=0),
                )
            context_bonus = 2 if context_topics and rule.topic in context_topics else 0
            if term_hits <= 0 and scenario_hits <= 0:
                if context_bonus <= 0:
                    continue
                if not detail_intents and len(question) > 12:
                    continue
            priority_bonus = 1 if rule.topic in _FORCED_TOPIC_PRIORITY else 0
            score = term_hits + scenario_hits * 2 + context_bonus
            if context_bonus > 0 and term_hits <= 0 and scenario_hits <= 0:
                score += min(len(detail_intents), 2)
            score += priority_bonus
            scored.append(
                MatchedRule(
                    rule=rule,
                    score=score,
                    specificity_bonus=specificity_bonus,
                    scenario_hits=scenario_hits,
                    term_hits=term_hits,
                    context_bonus=context_bonus,
                )
            )
        scored.sort(
            key=lambda item: (item.score, item.specificity_bonus, item.scenario_hits, item.context_bonus),
            reverse=True,
        )
        return scored

    def _select_rules(
        self,
        matched_entries: list[MatchedRule],
        *,
        question: str,
        detail_intents: list[str],
    ) -> list[MatchedRule]:
        if not matched_entries:
            return []
        selected = [matched_entries[0]]
        top = matched_entries[0]
        allow_parallel = any(term in question for term in ("，", ",", "以及", "并且", "同时", "还有", "另外"))
        for entry in matched_entries[1:]:
            if len(selected) >= 2:
                break
            if entry.rule.topic in {item.rule.topic for item in selected}:
                continue
            score_gap = top.score - entry.score
            if entry.explicit_hits > 0 and (score_gap <= 2 or (allow_parallel and score_gap <= 4)):
                selected.append(entry)
                continue
            if (
                entry.context_bonus > 0
                and entry.explicit_hits == 0
                and any(intent in detail_intents for intent in ("timeline", "fees", "materials", "contact", "eligibility"))
            ):
                selected.append(entry)
        return selected

    def _detect_detail_intents(self, question: str) -> list[str]:
        intents: list[str] = []
        for intent, terms in _DETAIL_INTENT_RULES:
            if any(term in question for term in terms):
                intents.append(intent)
        return intents

    def _match_scenario(self, rule: TopicRule, question: str) -> ScenarioRule | None:
        best_match: ScenarioRule | None = None
        best_score = 0
        for scenario in rule.scenarios:
            score = sum(1 for term in scenario.terms if term in question)
            if score > best_score:
                best_match = scenario
                best_score = score
        return best_match

    def _pick_field(self, rule: TopicRule, scenario: ScenarioRule | None, field_name: str) -> str:
        if scenario is not None:
            scenario_text = getattr(scenario, field_name)
            if scenario_text:
                return str(scenario_text)
        return str(getattr(rule, field_name))

    def _build_direct_conclusion(
        self,
        *,
        question: str,
        rule: TopicRule,
        scenario: ScenarioRule | None,
        detail_intents: list[str],
    ) -> str:
        scenario_name = scenario.name if scenario is not None else ""
        normalized = "".join(question.split())
        asks_timeline = "timeline" in detail_intents or any(
            term in normalized for term in ("多久到账", "多久能到", "多久能收到", "多久")
        )
        asks_fees = "fees" in detail_intents or any(
            term in normalized for term in ("运费", "费用", "收费", "多少钱", "原路返回", "信用卡")
        )
        asks_eligibility = "eligibility" in detail_intents or any(
            term in normalized for term in ("支持", "可以", "能否", "能不能", "还能", "影响")
        )

        if rule.topic == "refund_exchange":
            if scenario_name == "size_exchange":
                return "您好，支持更换尺寸。请在订单页发起换货申请，说明需要更换的目标尺寸；如涉及差价，按页面指引补差或退差即可。"
            if scenario_name == "refund_arrival":
                return "您好，退款会原路退回您的支付账户。一般1-3个工作日到账，信用卡支付可能需要3-5个工作日，具体以支付渠道处理速度为准。建议先确认订单是否已取消成功或售后审核通过，再查看退款状态。"
            if scenario_name == "seven_day_no_reason":
                return "您好，支持7天无理由退货。是否需要承担运费，通常取决于退换原因、商品是否存在质量问题以及平台售后规则。只要商品未使用、配件齐全，且仍在签收7天内，您可以在订单页直接申请退货，但运费需要您自行承担。"
            if scenario_name == "quality_reason":
                return "您好，商品存在质量问题支持退换货。请上传清晰的故障照片或视频，并提交质量问题售后申请，同时准备好订单号和问题说明。如核实属于质量问题，退换货运费由我们承担，建议越早提交越好。"
            if scenario_name == "opened_or_used":
                return "您好，已拆封商品如有质量问题（如污渍），可以按质量售后处理。建议您先保留订单信息、商品拆封状态和污渍的照片或视频作为证据，然后通过平台售后或投诉渠道提交申请。"
            if scenario_name == "refund_rejected":
                return "您好，退款被驳回后可以重新申请。请查看驳回原因，补齐相关证据后再次提交即可。"
            if asks_timeline or asks_fees:
                return "您好，退款会原路退回您的支付账户。一般1-3个工作日到账，信用卡支付可能需要3-5个工作日，具体以支付渠道处理速度为准。建议先确认订单是否已取消成功或售后是否审核通过，再查看退款状态。"
            if asks_eligibility:
                return "您好，未使用且在时效内的商品支持退换货。请在订单页发起售后申请。"
            return "您好，支持退换货。请在订单页发起售后申请，如需帮助可联系人工客服。"

        if rule.topic == "invoice":
            if scenario_name == "invoice_type" or any(
                term in normalized for term in ("发票类型", "专票", "普票", "电子发票", "纸质发票")
            ):
                return "您好，支持开具电子普通发票和增值税专用发票。请在订单页选择开票类型并填写抬头信息。"
            if scenario_name == "invoice_reissue":
                return "您好，发票抬头填错后可以申请更正或重开。如尚未开具，直接在订单页修改抬头即可；如已开具，请提供订单号和正确的公司名称、税号，我们会为您处理。"
            if scenario_name == "invoice_after_issued":
                return "您好，已开具的发票可以申请更正或重开。请提供订单号和正确的开票信息，我们会尽快处理。"
            return "您好，支持开发票。请在订单页查看开票入口，填写抬头和税号信息即可。"

        if rule.topic == "shipping":
            if scenario_name == "village_or_overseas":
                return "您好，大部分乡镇地区支持配送，运费与市区一致。偏远乡镇可能需要3-7天送达。"
            if scenario_name == "tracking_stalled":
                return "您好，物流长时间未更新可能是揽收延迟或中转滞留。建议您提供订单号，我们会立即为您催查。"
            if scenario_name == "package_lost_or_returned":
                return "您好，包裹异常我们会立即核查。请提供订单号和物流单号，我们会联系承运商确认包裹状态并为您处理。"
            if scenario_name == "signed_but_missing":
                return "您好，显示签收但未收到的情况，我们会立即联系承运商核实。请提供订单号，我们会为您跟进处理。"
            if scenario_name == "redirect_or_wrong_address":
                return "您好，还未签收的包裹可以尝试改派。请提供订单号和新地址，我们会联系承运商处理。"
            if any(term in normalized for term in ("乡镇", "国外", "海外", "寄到国外")):
                return "您好，大部分乡镇地区支持配送，运费与市区一致。偏远地区运费以页面显示为准。"
            if asks_fees:
                return "您好，正常配送运费与市区一致，不会额外加收。偏远地区运费以页面显示为准。"
            return "您好，正常情况下下单后24-48小时发货。如有物流问题请提供订单号，我们为您催查。"

        if rule.topic == "complaint":
            if any(term in normalized for term in ("假货", "不是正品", "验证是假")):
                return "您好，非常重视您关于商品真伪的反馈。请您提供订单号、商品页面宣传截图、实物照片以及验真凭证，我们会立即为您提交升级核查。投诉处理的时效要看证据完整度和平台工单流程，具体进度以平台反馈为准。"
            if any(term in normalized for term in ("虚假宣传", "宣传和实际不一样", "功能不符")):
                return "您好，非常重视您关于虚假宣传的反馈。请您提供订单号、商品页面宣传截图以及能证明实际功能不符的照片或视频，我们会为您提交升级处理。"
            if any(term in normalized for term in ("辱骂", "态度差", "态度特别差")):
                return "您好，非常重视您的反馈。请您提供订单号、相关时间及对话记录，我们会尽快核实并为您升级反馈。"
            if any(term in normalized for term in ("二手", "拆封", "污渍", "明显是二手")):
                return "您好，非常重视您的反馈。收到疑似二手商品的情况，请您立即拍摄商品状态照片和视频，保留外包装和物流面单，并通过平台投诉渠道或联系人工客服提交证据，我们会为您优先处理。"
            if any(term in normalized for term in ("保质期", "临近过期", "快过期", "过期")):
                return "您好，非常抱歉给您带来不好的体验。收到临近过期的商品，请您提供订单号和商品保质期照片，我们会尽快核实处理。"
            return "您好，非常重视您的反馈。请您提供订单号、相关证据（照片/视频/聊天记录），我们会尽快为您升级处理。"

        if rule.topic == "after_sales":
            if scenario_name == "in_warranty":
                return "您好，保修期内的商品支持免费维修。请提交售后申请并描述故障现象，我们会安排检测处理。"
            if scenario_name == "out_of_warranty":
                return "您好，超出保修期的商品可能需要付费维修。建议您先咨询售后确认维修方案和费用。"
            if scenario_name == "human_damage":
                return "您好，人为损坏通常不能按免费保修处理，但很多情况下仍可申请付费检测或付费维修。请提交售后申请。"
            if scenario_name == "repair_delay_or_repeat_failure":
                return "您好，非常抱歉给您带来不便。维修进度延迟我们会优先加急处理。请提供您的维修单号和送修时间，我们会立即为您核实当前维修状态并推进完成。如维修后短期内出现同样故障，支持免费重新维修。"
            if scenario_name == "rejected_or_disputed":
                return "您好，售后申请被驳回后可以申诉。请查看驳回原因并补充相关证据，我们会为您重新审核。"
            if asks_fees:
                return "您好，保修期内非人为损坏免费维修。超出保修期或人为损坏可能产生维修费用，具体以检测结果为准。"
            return "您好，支持售后维修服务。请提交售后申请并描述故障现象，我们会尽快为您安排。"

        if rule.topic == "quality_issue":
            if scenario_name == "packaging_damage":
                return "您好，收到外包装破损的商品，请先拍照留证，并仔细核对商品本体和配件是否完好。如有异常，请尽快在订单页发起售后申请并上传相关证据。"
            if scenario_name == "missing_items":
                return "您好，少发漏发支持补寄。请提供订单号和缺少的配件信息，我们会核实并尽快安排补发。如果之前已提供信息但未处理，建议联系人工客服跟进。"
            if any(term in normalized for term in ("过期", "临期", "保质期")):
                return "您好，非常抱歉给您带来不好的体验。收到临近过期的商品，请您提供订单号和商品保质期照片，我们会尽快核实处理。如商品在保质期内但临近过期，且下单时页面未标注临期，您可以申请退货退款，运费由我方承担。"
            return "您好，请尽快拍照留证并在订单页发起售后申请，我们会为您处理。"

        if rule.topic == "order_change":
            if scenario_name == "refund_arrival":
                return "您好，退款会原路退回您的支付账户。一般1-3个工作日到账，信用卡可能需要3-5个工作日。"
            if "取消订单" in normalized:
                return "您好，未发货的订单可以直接取消。已发货的订单请拒收后申请退款。"
            return "您好，未发货的订单可以直接取消。已发货的请拒收后申请退款，退款会原路退回。"

        if rule.topic == "manual_request":
            return "您好，电子版说明书可通过商品页或品牌官网获取。如需纸质版补寄，请联系客服确认。"

        if rule.topic == "platform_service":
            if scenario_name == "trade_in":
                return "您好，支持以旧换新服务。请在商品页查看是否有以旧换新入口，按页面要求填写旧机信息即可。"
            if scenario_name == "coupon":
                return "您好，优惠券在满足有效期和使用门槛后，可在结算页直接勾选使用。"
            if scenario_name == "trial_sample":
                return "您好，部分商品支持试用服务。请在商品页查看是否有试用入口。"
            if scenario_name == "trial_extension_or_fault":
                return "您好，试用期间如遇故障，请尽快提交故障证据和延期诉求，我们会为您协调处理。"
            if scenario_name == "smart_customer_service":
                return "您好，智能客服可处理常见订单、物流、售后和发票问题。如需人工客服，我帮您转接。"
            return "您好，相关服务请以商品页和活动页入口为准。如有疑问可联系人工客服。"

        if rule.topic == "address_change":
            if scenario_name == "address_after_shipment":
                return "您好，已发货的订单改地址需要联系承运商处理。请提供订单号和新地址，我们会为您协调。"
            return "您好，未出库的订单可以直接修改地址。已发货的请联系客服协助处理。"

        if rule.topic == "installation_service":
            if scenario_name == "installation_reschedule":
                return "您好，已预约的安装服务可以改约。请在服务单里修改时间，或联系客服重新安排。"
            return "您好，部分商品支持预约安装服务。请在订单页查看安装服务入口。"

        if rule.topic == "price_protection":
            return "您好，支持价保服务。请在订单页提交价保申请，我们会为您核实处理。"

        if rule.topic == "payment_issue":
            if scenario_name == "repeat_charge":
                return "您好，重复扣款我们会立即核查。请提供订单号和扣款截图，我们会为您确认并处理退款。"
            return "您好，支付异常请提供订单号和扣款截图，我们会尽快为您核查处理。"

        if rule.topic == "delivery_delay":
            return "您好，已为您催促发货。如超过承诺时效仍未发货，我们会为您跟进处理。"

        if rule.topic == "warranty_period":
            return "您好，保修期一般为购买之日起1年，具体以商品页和保修卡为准。"

        if rule.topic == "accessory_request":
            if scenario_name == "accessory_rejected":
                return "您好，补寄被驳回后可以重新申请。请补充缺件证据（开箱视频/照片），我们会为您重新审核。"
            return "您好，缺少配件支持补寄。请提供订单号和缺少的配件名称，我们会尽快为您安排。"

        return self._pick_field(rule, scenario, "overview")

    def _compose_topic_sections(
        self,
        rule: TopicRule,
        *,
        question: str,
        detail_intents: list[str],
        scenario: ScenarioRule | None,
    ) -> dict[str, list[str]]:
        overview = self._pick_field(rule, scenario, "overview")
        materials = self._pick_field(rule, scenario, "materials")
        timeline = self._pick_field(rule, scenario, "timeline")
        fees = self._pick_field(rule, scenario, "fees")
        eligibility = self._pick_field(rule, scenario, "eligibility")
        process = self._pick_field(rule, scenario, "process")
        contact = self._pick_field(rule, scenario, "contact")

        conclusions = [
            self._build_direct_conclusion(
                question=question,
                rule=rule,
                scenario=scenario,
                detail_intents=detail_intents,
            )
        ]
        if "eligibility" in detail_intents:
            conclusions.append(eligibility)
        elif scenario is not None and overview not in conclusions:
            conclusions.append(overview)

        steps = [process]
        if "materials" in detail_intents or not detail_intents:
            steps.append(materials)

        timing_fee: list[str] = []
        if "timeline" in detail_intents:
            timing_fee.append(timeline)
        if "fees" in detail_intents:
            timing_fee.append(fees)
        if "eligibility" in detail_intents:
            timing_fee.append(eligibility)

        return {
            "conclusion": self._dedupe_items(conclusions),
            "steps": self._dedupe_items(steps),
            "timing_fee": self._dedupe_items(timing_fee),
            "support": self._dedupe_items([contact] if "contact" in detail_intents else []),
        }

    def _compose_answer(
        self,
        *,
        question: str,
        sections: list[dict[str, list[str]]],
        matched_topics: list[str],
        detail_intents: list[str],
    ) -> str:
        conclusions = self._merge_section_items(sections, "conclusion")
        steps = self._merge_section_items(sections, "steps")
        timing_fee = self._merge_section_items(sections, "timing_fee")
        support = self._merge_section_items(sections, "support")

        # For simple questions without specific intents, just return the conclusion directly
        if not detail_intents and conclusions:
            return conclusions[0]

        reassurance = self._build_reassurance_line(
            question,
            matched_topics,
            detail_intents=detail_intents,
            has_support=bool(support),
        )
        if reassurance:
            support.append(reassurance)
        support = self._dedupe_items(support)

        blocks: list[str] = []
        if conclusions:
            blocks.append("结论：\n- " + "\n- ".join(conclusions[:2]))
        if steps:
            blocks.append(
                "处理步骤：\n" + "\n".join(
                    f"{index}. {item}" for index, item in enumerate(steps[:3], start=1)
                )
            )
        if timing_fee:
            blocks.append("时效/费用：\n- " + "\n- ".join(timing_fee[:3]))
        if support:
            blocks.append("补充说明：\n- " + "\n- ".join(support[:2]))
        return "\n\n".join(blocks).strip()

    def _merge_section_items(
        self,
        sections: list[dict[str, list[str]]],
        key: str,
    ) -> list[str]:
        items: list[str] = []
        for section in sections:
            items.extend(section.get(key, []))
        return self._dedupe_items(items)

    def _dedupe_items(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for raw_item in items:
            item = " ".join(str(raw_item).split()).strip()
            if not item or item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    def _build_reassurance_line(
        self,
        question: str,
        matched_topics: list[str],
        *,
        detail_intents: list[str],
        has_support: bool,
    ) -> str:
        asks_materials_or_contact = any(intent in detail_intents for intent in ("materials", "contact"))
        asks_customer_service_handoff = any(term in question for term in ("客服", "人工", "联系谁", "怎么联系", "转人工"))
        if "complaint" in matched_topics:
            return "如果已经影响你的使用体验或服务体验，建议尽快固定截图、聊天记录和页面证据，再升级到人工客服或平台投诉渠道。"
        if not asks_materials_or_contact and not asks_customer_service_handoff:
            return ""
        if has_support:
            return ""
        if "shipping" in matched_topics or "delivery_delay" in matched_topics or "quality_issue" in matched_topics:
            return "如果已经影响签收、使用或补寄时效，建议尽快带上订单号和截图联系人工客服加急核查。"
        if "after_sales" in matched_topics or "refund_exchange" in matched_topics:
            return "如果问题已经影响正常使用，建议一次性准备订单号、故障照片或售后截图，这样通常更快进入人工审核。"
        if "invoice" in matched_topics:
            return "如果你这边已经有订单号和开票截图，建议一并提供，通常可以减少重复核对。"
        if "客服" in question or "人工" in question:
            return "如果智能客服无法直接解决，建议转人工时一次性说明诉求、订单号和异常截图。"
        return ""
